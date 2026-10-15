# gb10-rendering (ComfyUI, Qwen-Image-2512) — notes

**Status (2026-10-05): smoke-tested successfully, first try.** 512x512 test
generation via `POST /generate` returned a correct photorealistic portrait
in 21s (HTTP 200), **while `qwen3.8-27b-uc-nvfp4` + `bge-m3` were both still
running** — confirms the parallel-operation claim holds at least at this
resolution (not yet tested at higher resolutions/batch, where activation
memory grows). The LoRA stack loaded correctly ("patches attached" in the
ComfyUI log). Output directory
confirmed empty afterward — the delete-on-handoff cleanup works. The
hand-built `workflow-text2image.json` graph worked on the first attempt, no
node-type corrections needed.
Goal: photorealistic text-to-image generation, callable by agents on the
harness server (text in, image out — the agent side handles posting the
result to a chat channel, that's not built here).

**Renderers are configurable.** The public repo ships two example renderers
(`renderers.json`: `qwen` = Qwen-Image-2512 + Lightning, `flux` = FLUX.1-dev).
A box can replace that registry with its own in the git-ignored
`comfyui/local/` (renderer ids, workflows, LoRA file names, its own notes) —
see "Renderer registry + optional LoRAs" below.

## Why this is a separate thing from everything else in `~/llm`

Qwen-Image is a diffusion transformer (`diffusers`/ComfyUI world), not an
LLM — vLLM cannot serve it. None of `common/lib.sh`/`common/serve.sh`
applies (that's vLLM-CLI-flag building specifically). This directory is a
self-contained ComfyUI setup: its own `Dockerfile`, `serve.sh`,
`install-service.sh` (separate systemd unit, not an instance of the shared
`llm-vllm@.service` template), and API workflow JSON.

## Model stack

Switched from the original plan's "pre-merged FP8+Lightning single file"
(lightx2v's `qwen_image_2512_fp8_e4m3fn_scaled_4steps_v1.0.safetensors`) to
**separately-stacked LoRAs on a first-party Comfy-Org base checkpoint**,
found mid-implementation — better provenance (Comfy-Org is ComfyUI's own
org, guaranteed-correct folder/filename conventions) and more flexible
(toggle the Lightning LoRA or stack further LoRAs independently, just by
editing the workflow JSON, no need to re-download a different merged file).

| file | source | size | →  ComfyUI folder |
|---|---|---|---|
| `qwen_image_2512_fp8_e4m3fn.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` (first-party) | 20.43 GiB | `models/diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` (first-party) | 9.38 GiB | `models/text_encoders/` |
| `qwen_image_vae.safetensors` | `Comfy-Org/Qwen-Image_ComfyUI` (first-party) | 0.254 GiB | `models/vae/` |
| `Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors` | `lightx2v/Qwen-Image-2512-Lightning` (973K downloads) | 0.85 GiB | `models/loras/` |


Total ~31 GiB resident when loaded.

**Text encoder note**: Comfy-Org's `Qwen-Image_ComfyUI` repo names this file
generically (`qwen_2.5_vl_7b_*`, no version suffix) and reuses it across
`qwen_image`/`qwen_image_2512`/etc. diffusion-model variants in the SAME
repo — confirms the encoder is shared across Qwen-Image releases, not
version-locked. A smaller `qwen_2.5_vl_7b_nvfp4.safetensors` (6.11 GiB) also
exists in that repo — **not used yet**, unverified whether ComfyUI's CLIP
-loading path has working NVFP4 kernel support on this box's sm_121/Blackwell
GPU (vLLM's NVFP4 support doesn't automatically mean ComfyUI's separate
inference backend does too). Worth trying later if VRAM headroom turns out
too tight — would save ~3.3 GiB vs. the FP8 scaled option used now.

## Second renderer: FLUX.1-dev (added 2026-10-06)

FLUX.1-dev (or a finetune of it) as an additional renderer next to
Qwen-Image. Selected per request via bridge.py:
`POST /generate {"prompt": ..., "model": "flux"}` (default stays `"qwen"`,
so existing callers are unaffected). Graph:
`workflow-flux-text2image.json` (25 steps, euler/simple, FluxGuidance 3.5,
cfg 1.0 -- FLUX.1-dev is guidance-distilled, no negative prompt).

| file | source | size | → ComfyUI folder |
|---|---|---|---|
| `flux1-dev_fp8_e4m3fn.safetensors` | FLUX.1-dev transformer, fp8 (or a finetune converted with `convert-flux-diffusers.py`, see below) | ~11.1 GiB | `models/diffusion_models/` |
| `ae.safetensors` | FLUX.1-dev VAE (a diffusers-layout `vae/` works too; ComfyUI's VAELoader converts diffusers VAE keys itself) | 0.16 GiB | `models/vae/` |
| `clip_l.safetensors` | `comfyanonymous/flux_text_encoders` (first-party) | 0.23 GiB | `models/text_encoders/` |
| `t5xxl_fp8_e4m3fn_scaled.safetensors` | `comfyanonymous/flux_text_encoders` (first-party) | 4.8 GiB | `models/text_encoders/` |

**Finetunes in diffusers layout**: some FLUX finetunes on HF only ship the
diffusers layout (sharded, bf16). `convert-flux-diffusers.py` (runs inside the comfyui
image, uses ComfyUI's own `flux_to_diffusers` key map) fuses it into one
BFL-key file and casts the block linear weights to fp8_e4m3fn (embedders,
norms, biases stay bf16 -- same split as the usual flux1-dev-fp8 releases).
It streams tensor by tensor, because a load-all conversion needs more RAM
than is free next to the vLLM models. Staging download lives in
`~/models/comfyui-cache/staging/` (outside the models mount) and is deleted
after a successful conversion.

**Text encoders**: for a transformer-only finetune the stock CLIP-L/T5-XXL
that FLUX.1-dev ships with are the right ones. Swap in the repo's encoders if prompts behave oddly.

**Smoke test (2026-10-06), with qwen3.8-27b-uc-nvfp4 + bge-m3 running**:
flux 512x512 26s (incl. cold model load), flux 1024x1024 32s, then qwen
512x512 23s (switch back, incl. reload). All HTTP 200, output dir empty
afterwards. ComfyUI logged "0 models unloaded" on the switch back -- both
stacks stayed resident and it still fit (MemAvailable ~21 GiB after). Not
yet tested: qwen at 1024+ while flux is still resident -- if that ever
OOMs, start ComfyUI with `--disable-smart-memory` (always unloads after use).
Bridge timeout raised to 600s for cold loads.

**Default renderer / switching (added same day)**: requests without a
`"model"` field use `~/models/comfyui-cache/bridge/default-model`
(`qwen`|`flux`, mounted ro at `/opt/bridge`, re-read per request), toggled
with `i` in `llmctl`. When the model differs from the previous request's,
bridge.py POSTs `/free {"unload_models": true}` and waits 3s before queueing
(the flag is only handled between queue items). Verified: qwen -> flux
switch dropped host "used" from 102.4 to 91.6 GB, i.e. qwen really got
unloaded instead of both stacks staying resident.

## Renderer registry + optional LoRAs (2026-10-15)

`renderers.json` maps a request's `"model"` id to a workflow template, a
weight stack (`qwen` | `flux` — selects the preset block and whether a
switch unloads the other stack) and a display label for llmctl. A box-local
`comfyui/local/renderers.json` (git-ignored; `serve.sh` mounts the dir
read-only at `/opt/bridge-local` when it exists) replaces the public file
entirely, with workflow paths relative to it — that is where box-specific
renderers, model files and their notes live.

Two optional LoRA hooks, both configured in the registry:
- `"lora_node"` on a renderer: id of a `LoraLoaderModelOnly` node in its
  workflow; the request field `lora_strength` (0–2, default 1.0) sets it.
- top-level `"extra_lora"`: `{"file": <models/loras/ file>, "stacks": [...]}`;
  a request with `lora2` (`true` = 1.0 / number 0–2 / `false`, default off)
  splices it in after whatever feeds the sampler (node 40), so hires and
  detailers inherit it. Renderers on other stacks ignore it.

Switching between renderers of the same stack (e.g. a base model and the
same base with a LoRA) does not unload.

## Negative prompt (added 2026-10-07)

Request field `negative` (flux stack; qwen ignores it). It goes into the
already-wired negative CLIPTextEncode (node 7) plus a second FluxGuidance
(node 15, same guidance value as the positive side). A negative prompt only
does anything with real CFG > 1 -- FLUX-dev is distilled to cfg 1, where
ComfyUI skips the negative branch -- so a request with `negative` and no
`cfg` gets **cfg 3.0** (`NEGATIVE_DEFAULT_CFG`). Costs ~1.5x render time
(fast preset 768²: 18 s -> 28 s). Writing "negative: ..." into the prompt
itself does the opposite (T5 reads it as content). The text is never logged;
`X-Render-Settings` only carries `"negative": true`.

Test (flux, fast, seed 5, "bowl of mixed fruit", negative "bananas,
apples"): cfg 1 -> bowl full of apples; cfg 2 -> a few green apples left;
cfg 3 -> none, natural look; cfg 4 -> none, slightly oversaturated. Against
a positive prompt that explicitly asks for the thing (street "with parked
cars" vs. negative "cars") it only reduces it -- the positive wins.

Also fixed: `"seed": null` used to crash with HTTP 500, now means random.

## Repeat requests + PNG metadata (fixed 2026-10-07)

- **Identical repeat request → HTTP 500** ("No such file … output/
  gb10-rendering_00001_.png"): ComfyUI skips every node whose inputs are
  unchanged, so a byte-identical graph (same prompt, seed, settings) never
  re-ran SaveImage and reported the previous run's file, which the bridge had
  already deleted ("Prompt executed in 0.00 seconds" in the log). Fix: the
  bridge gives SaveImage a unique `filename_prefix` per request — only
  SaveImage re-runs, the image comes from the in-memory cache (~1 s instead
  of a full render), output is byte-identical.
- **Prompt inside the PNG**: SaveImage embeds the whole graph (prompt and
  negative text included) as PNG tEXt chunks by default — the prompt would
  have travelled inside every image (a chat app's *document* upload keeps it).
  ComfyUI now runs with `--disable-metadata`; verified: no tEXt chunks,
  prompt string not in the file.

## Timeout + abort cleanup (fixed 2026-10-10)

- `GENERATE_TIMEOUT_S` 900 -> **1800 s** (env-overridable, `serve.sh` passes
  it through: `GENERATE_TIMEOUT_S=20 comfyui/serve.sh start` for a test). It
  counts from submission, i.e. includes queue wait behind other requests,
  and everything is slower while vLLM is busy. **Clients (the agent harness,
  render.py) need a longer timeout than this** (render.py: 1900 s).
- **No-artifact hole closed**: on timeout the bridge used to just stop
  waiting -- the job kept running in ComfyUI and later wrote its PNG to
  output/ plus the prompt to /history with nobody left to delete them. Now
  every path out of `generate()` (success, error, timeout, cancellation)
  runs `abort_job()` if the job isn't finished -- `POST /queue {delete}` for
  a pending job, `POST /interrupt` only while `queue_running` shows *our*
  prompt_id -- waits until ComfyUI has let go of it, then `cleanup_job()`
  deletes every file with the request's unique prefix and its history entry.
- Startup sweep: bridge empties output/ + temp/ and clears /history once at
  startup (safety net after a crash/kill mid-job). The first run removed **3
  leftover files** -- most likely exactly such timed-out jobs from before.
- Verified (timeout 20 s, preset final 1024²): HTTP 500 after 20 s, job
  interrupted (ComfyUI "executed in 20.27 s" instead of ~260 s), queue,
  history, output/ and temp/ empty, follow-up request served right away;
  same with two parallel requests (one running, one pending).
- A client that disconnects doesn't abort anything: aiohttp doesn't cancel
  the handler, the job finishes and is cleaned up as usual.

## Presets / render parameters (added 2026-10-07)

`comfyui/presets.json` holds four presets x two stacks; bridge.py resolves
request field > preset (request `preset` > `bridge/default-preset`, toggled
with `p` in llmctl > `realistic`). Request fields: `preset, steps, cfg,
guidance, sampler, scheduler, hires, face_fix, hand_fix` (see bridge.py
docstring). qwen ignores steps/cfg/guidance/sampler/scheduler (Lightning).

**How common A1111/SD1.5 rules of thumb map here** ("CFG 5-7 realistic,
DPM++ 2M Karras, 25-30 / 40-50 steps, hires 2x R-ESRGAN denoise 0.4,
ADetailer"):
- CFG -> FLUX-dev is guidance-distilled: real `cfg` stays 1.0 (>1 doubles
  time, degrades), the strength knob is `guidance` (FluxGuidance).
- Karras -> **measurably worse on FLUX** (A/B grid, FLUX + LoRA, 768², seed
  42): karras (dpmpp_2m and dpmpp_sde) gave washed-out images, melted cup,
  merged fingers; simple/beta/sgm_uniform all crisp. Picked dpmpp_2m + beta.
  dpmpp_sde is also 2x slower (94s vs 48s).
- guidance sweep 2.0/2.5/2.8/3.5 with dpmpp_2m+beta: 2.5 most natural
  (2.0 slightly flat, >=2.8 glossier). Higher guidance also amplifies a
  stacked LoRA's style bias, so keep it moderate with LoRAs.
- Steps: realistic 28, final 40, fast 20 (euler+beta).
- Hires: RealESRGAN_x4plus (official xinntao v0.1.0 release, sha256
  4fa0d389...) -> scaled to 1.5x (not 2x: FLUX at 2048² is very expensive
  and chat apps downsize photos to ~1280 anyway), denoise 0.4, 20 steps.
- ADetailer -> Impact Pack 8.28 FaceDetailer + Impact Subpack 1.3.4
  UltralyticsDetectorProvider, detectors `Bingsu/adetailer` face_yolov8m /
  hand_yolov8s (ADetailer's own models, sha256-checked; .pt = pickle, but
  they loaded via Subpack's weights_only path, whitelist stays empty).
  guide_size 768 (flux) / 512 (qwen), denoise 0.4, 20 steps (qwen 4).
- `anime` preset (guidance 4.5, euler_ancestral) is untested beyond the
  schema -- with a realism finetune, expect limited anime results.

**Timings** (FLUX + LoRA 768², while vLLM was serving a request in parallel,
GPU shared): fast 43s, realistic (face fix) 96s, final (hires 1152² + face
+ hand fix) 262s; qwen realistic + face fix 59s. Plain FLUX without vLLM
load was ~24s/768² earlier, so expect roughly half when the LLM is idle.

**Image build**: Impact deps installed with a constraints file pinning the
NGC torch/torchvision/numpy/triton/transformers/scipy/pillow (ultralytics
would otherwise pull generic wheels); full opencv-python replaced by the
headless build; ultralytics telemetry off (`YOLO_OFFLINE=1`, `sync=False`).
No-artifact check: output/ and ComfyUI temp/ empty after every stage test --
detailer/hires intermediates stay in memory.

## Version choice: 2512 over the newer 2.1

`Qwen/Qwen-Image-2.1` (Sep 2026) is chronologically newer than
`Qwen-Image-2512` (Dec 2025), but has **no Lightning LoRA yet** (checked
this session — zero search hits) and an unclear `license:other` vs. 2512's
Apache 2.0. 2512 has a mature, fully-compatible stack (first-party FP8 +
popular Lightning LoRA), so picked that over the
technically-newer release.

## GPU memory risk — untested claim, verify before trusting

Qwen-Image-2512's ~31.5 GiB resident footprint, plus diffusion activation
memory (scales with resolution/batch, not fixed), runs **in parallel** with
whatever vLLM models are active (explicit design choice). With `qwen3.8-27b-nvfp4`(0.52)+`bge-m3`(0.06)
already reserving ~70.5 GiB of the ~121.6 GiB real pool, headroom for
ComfyUI + its activations is roughly 14-20 GiB — tight, same failure class
(OOM → Xid/cuBLAS/device-assert) as this box's two prior documented OOM
incidents (see `models/qwen3.8-27b-fp8/notes.md`, "2026-09-10"). **Test cautiously**:
first generation should be low-resolution (e.g. 512x512) with the LLMs
still running, before trusting larger batches/resolutions in that
configuration. If it OOMs, the fallback is treating this as on-demand-only
(stop an LLM first) rather than true parallel operation.

## ComfyUI version / native support

Pinned `v0.38.0` (latest at build time). Native Qwen-Image support landed
~v0.37.0 — no custom node pack needed. 0.38.0 adds Qwen-Image-2.1-specific
optimizations (block compile, KV cache, tiny VAE) that don't apply to 2512,
harmless either way since 2512 uses the generic Qwen-Image loading path.

## Workflow JSON — unverified node graph, confirm against a live instance

`comfyui/workflow-text2image.json` was hand-built from documented ComfyUI
conventions (`UNETLoader` for `diffusion_models/`, `CLIPLoader` with
`type: qwen_image` for the text encoder, `LoraLoaderModelOnly` for
the Lightning LoRA, `EmptySD3LatentImage` for the initial latent --
Qwen-Image uses the same latent format as SD3/Flux-style models in
ComfyUI). **Not yet validated against a running ComfyUI instance** -- once
it's up, check `GET /object_info` for the actual available node types and
their exact input names before trusting this graph, and prefer loading the
official example workflow (embedded in the PNG at
`comfyanonymous/ComfyUI_examples/qwen_image/qwen_image_basic_example.png`)
as a cross-check, then layer the `LoraLoaderModelOnly` node(s) onto that
known-good graph rather than trusting my hand-built one blind.

## No-artifact requirement (added after initial build, changed the architecture)

User requirement, stated explicitly: once a generated image has been handed
off and posted to the chat channel, **no artifact** (prompt text, image
file) may remain — neither on the LLM server nor on the harness server.

This changed the design from "the agent talks to raw ComfyUI's `/prompt` +
`/view` API" (the original plan) to a **bridge service** (`bridge.py`) that
is the *only* thing published on port 8002:

- Raw ComfyUI binds `127.0.0.1:8188` inside the container only — launched
  as `bridge.py`'s subprocess, never reachable from the LAN. This is
  deliberate: raw ComfyUI's API has no cleanup guarantee (the output file
  and the prompt text in `/history` just sit there after a generation), so
  it must never be the externally-reachable surface.
- `bridge.py` exposes `POST /generate` (`{"prompt": ..., "width": 1024,
  "height": 1024, "seed": optional}` → `image/png` bytes directly in the
  response body) and `GET /health`. Internally it: builds the graph from
  `workflow-text2image.json`, submits to `127.0.0.1:8188/prompt`, polls
  `/history/<id>`, reads the output PNG, **deletes that file** and **POSTs
  `{"delete": [prompt_id]}` to ComfyUI's `/history`** to clear the stored
  prompt text — all before returning the bytes to the caller. Cleanup runs
  in a `finally` block, so it happens even if something upstream errored.
- **LLM-server-side guarantee**: solid — the file and history entry are gone by
  the time the HTTP response is sent, unconditionally (not gated on the
  caller's later chat post succeeding; this box has no way to know that
  outcome and shouldn't hold a file hostage waiting for a confirmation that
  might never arrive).
- **Harness-side guarantee**: NOT enforceable from here. Whatever implements the
  agent tool-call for this must receive the PNG bytes from `/generate` and
  hand them straight to the chat upload call **without writing them to
  disk** (or deleting immediately after if the chat client library
  requires a file path) — document this requirement wherever that agent
  tool is built, it's outside this repo/box entirely.
- **Known gap, not chased further**: ComfyUI's own asset-tracking SQLite
  database (added in recent ComfyUI releases for the asset browser) may
  retain a metadata row referencing the now-deleted file/history entry.
  This is ComfyUI's internal bookkeeping, not something `bridge.py`
  currently reaches into — the actual prompt TEXT and image BYTES are gone
  (the two things that actually matter for this requirement), but a
  filename/timestamp stub could theoretically linger in that DB. Revisit if
  this turns out to matter.

`workflow-text2image.json` is now consumed internally by `bridge.py`, not
POSTed directly by the agent harness.

## Firewall / access

Port `8002`. Whitelisted for the **harness server** (agent calls) and the **workstation**
(interactive workflow building in ComfyUI's own UI) via
`common/setup-firewall.sh` (DOCKER-USER chain, same mechanism as the other
dockerized ports — plain `ufw allow` doesn't work for Docker-published
ports, see that script's header). No other host has access, same as
everywhere else on this box.

## Deliberately NOT done (scope decisions)

- **Not boot-enabled.** `install-service.sh` installs but does not
  `enable`/`start` the unit — given the memory-risk above, starting this
  automatically on every boot alongside the LLM boot-defaults isn't a
  reasonable default; start it manually when wanted.
- **No chat-posting code on this box.** The request (agents should call it
  and post the result to a chat channel) reads as the agent side handling
  that leg itself; this box only needs to expose the
  image-gen API. Flag/revisit if that assumption is wrong.
