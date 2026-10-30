# LLM + image-gen serving on the DGX Spark (GB10)

vLLM serving + benchmark harness for LLMs, plus a ComfyUI-based image
-generation service (`comfyui/`, a different engine entirely), on a single
**NVIDIA GB10** (DGX Spark, 128 GB unified LPDDR5x, sm_121, CUDA 13).

One generic launcher in `common/`, one directory per model in `models/`.
Adding a model = a new `models/<slug>/` with a `model.env` (an embedding model
just adds `RUNNER=pooling` — see `bge-m3/`).

```
common/
├── box.env             hardware facts: NGC image, caches, sm_121 quirks
├── serve.sh            generic launcher   (serve.sh <model-dir> <subcommand>)
├── lib.sh              shared implementation
├── llm-vllm@.service   systemd template (reference; install-service generates the real one)
├── setup-firewall.sh   ufw + DOCKER-USER whitelist (sudo; --dry-run prints the rules)
├── firewall.local.env.example   template for the git-ignored firewall.local.env (client IPs)
└── bench/              run.py · report.py · ab_lmonly.py · _env.py
models/
├── qwen3.8-27b-fp8/    FP8 vision MoE, MTP speculative decoding, uncensored   (benchmark-tuned)
│   ├── model.env       identity + benchmark-tuned serving knobs   ← the parametrization
│   ├── notes.md        model-card facts + gotchas
│   ├── serve           thin wrapper -> common/serve.sh with this dir
│   ├── docker/         Dockerfile.xgrammar024 (xgrammar 0.2.0->0.2.4 patch, see notes.md)
│   └── bench/
│       ├── matrix.yaml   the sweep (configs, workloads, suites)
│       ├── ocr_test.png
│       └── results/      per-run JSON, server logs, summary.csv, report.md/html, plots
├── qwen3.8-27b-nvfp4/  official NVIDIA NVFP4, stock (non-abliterated)   (bench-swept)
│   ├── model.env       benchmark-tuned knobs, see notes.md sweep results
│   ├── notes.md        full bench sweep vs. FP8, GDN/Triton workaround, kernel selection
│   ├── serve           thin wrapper -> common/serve.sh with this dir
│   └── bench/results/  gap_close_nvfp4 sweep
├── qwen3.8-27b-uc-nvfp4/  orcarouter uncensored NVFP4   (added, not yet smoke-tested)
│   ├── model.env       inherited starting-point knobs from qwen3.8-27b-nvfp4, unverified here
│   ├── notes.md        lineage, compressed-tensors quant encoding (differs from qwen3.8-27b-nvfp4's ModelOpt), open questions
│   └── serve           thin wrapper -> common/serve.sh with this dir
├── gpt-oss-120b/       native MXFP4 text MoE, Harmony format   (benchmark-tuned)
│   ├── model.env       same layout as above, all knobs benchmark-tuned
│   ├── notes.md         model-card facts, SM121 MXFP4 bug (verified NOT reproducing), sweep results
│   ├── serve            thin wrapper -> common/serve.sh with this dir
│   └── bench/
│       ├── matrix.yaml   the sweep (configs, workloads, suites)
│       └── results/      per-run JSON, server logs, summary.csv (4 suites: backend/seqs/kv/context)
├── gemma4-26b-heretic-nvfp4/  Gemma 4 MoE NVFP4, uncensored, 262144 ctx   (gb10-writing, parallel to qwen)
│   ├── model.env       knobs + patched-image override; thinking on by default
│   ├── notes.md        why, tied-lm_head fix, writer comparison (qwen / Small 4 / Gemma)
│   ├── serve           thin wrapper -> common/serve.sh with this dir
│   └── docker/         Dockerfile.tiedhead + patch for vLLM 0.24's ModelOpt tied-head bug
└── bge-m3/            XLM-RoBERTa encoder, 1024-dim multilingual embeddings   (RUNNER=pooling)
    ├── model.env       identity + pooling knobs (no chat/generate flags)
    ├── notes.md         dense/sparse/ColBERT facts, --runner pooling, fallbacks
    └── serve            thin wrapper -> common/serve.sh with this dir
comfyui/                gb10-rendering: Qwen-Image-2512 text-to-image, ComfyUI   (different engine, NOT vLLM)
├── Dockerfile           NGC PyTorch 26.07 (arm64) + ComfyUI v0.38.0 + bridge.py; ONLY bridge.py is exposed
├── bridge.py            the published API (POST /generate, GET /health) -- generates, reads bytes, deletes
│                         the output file + ComfyUI history entry before responding (no-artifact requirement)
├── serve.sh             launcher (start|run|stop|logs|status) -- not common/serve.sh, different engine
├── install-service.sh   installs comfyui.service (separate unit, not llm-vllm@) -- needs plain sudo
├── renderers.json       renderer registry: model id -> workflow, stack, label, optional LoRA hooks
├── workflow-text2image.json        Qwen-Image-2512 + Lightning 4-step graph template
├── workflow-flux-text2image.json   FLUX.1-dev graph template (both used internally by bridge.py,
│                         not POSTed by callers)
├── presets.json         render presets (realistic / final / fast / anime) per stack
├── local/               git-ignored: box-specific renderer registry, workflows, LoRA names, notes
└── notes.md             model choice, memory-risk, no-artifact design, renderer registry
llmctl                  interactive terminal dashboard: toggle models, memory-fit check
.env                    box-wide secrets: HF_TOKEN
.hf-venv/               venv for the bench harness + hf downloads
```

Weights and the vLLM compile cache live outside the repo, by default
`/home/ss/models/hf-cache` and `/home/ss/models/vllm-cache` (`HF_CACHE` /
`VLLM_CACHE` in `common/box.env`; ComfyUI: `COMFYUI_CACHE`, default
`/home/ss/models/comfyui-cache`).

## Models on this box

| slug | served as | port | endpoint | context | license | status |
|---|---|---|---|---|---|---|
| `qwen3.8-27b-uc-nvfp4` | `gb10-reasoning` | 8000 | `/v1/chat/completions` | 262144 | Apache 2.0, uncensored | smoke-tested, current `gb10-reasoning` boot-default (systemd) |
| `qwen3.8-27b-nvfp4` | `gb10-reasoning` | 8000 | `/v1/chat/completions` | 262144 | Apache 2.0, stock | bench-swept, on-demand alternate |
| `qwen3.8-27b-fp8` | `gb10-reasoning` | 8000 | `/v1/chat/completions` | 131072 | Apache 2.0, uncensored | benchmark-tuned, on-demand alternate |
| `gpt-oss-120b` | `gb10-reasoning` | 8000 | `/v1/chat/completions` | 65536 | Apache 2.0, stock safety | benchmark-tuned, on-demand alternate |
| `bge-m3` | `gb10-embedding` | 8001 | `/v1/embeddings` | 8192 | MIT | in use, embeddings behind an OpenAI-compatible proxy (systemd, boot) |
| `comfyui` (`gb10-rendering`) | `gb10-rendering` | 8002 | `/generate` (bridge API) | n/a (image gen) | qwen: Apache 2.0 + Lightning LoRA; flux: FLUX.1 [dev] non-commercial | smoke-tested, boot-enabled (systemd) |
| `gemma4-26b-heretic-nvfp4` | `gb10-writing` | 8004 | `/v1/chat/completions` | 262144 | Apache 2.0, uncensored (Heretic) | smoke-tested, boot-enabled (systemd), German stories / roleplay + non-coding agent profiles, runs **in parallel** with qwen (patched image, see its notes) |

All vLLM models share the 128 GB unified pool — see **Running multiple
models** below. They run under the `llm-vllm@` systemd template (`llmctl` or
`systemctl start|stop`); each `gb10-*` identity's members are mutually
exclusive (never co-resident, same port) except where noted. Currently
boot-enabled (autostart): `qwen3.8-27b-uc-nvfp4` + `bge-m3` + `gemma4-26b-heretic-nvfp4` (`gb10-writing`) + `comfyui`
(`gb10-rendering`, separate `comfyui.service` unit — see `comfyui/
install-service.sh`). `gb10-rendering` is a different engine entirely (not
vLLM, no shared pool budget carve-out) and is explicitly allowed to run **in
parallel** with whichever `gb10-reasoning` member is active —
confirmed working at 512x512 alongside `qwen3.8-27b-uc-nvfp4` + `bge-m3`
(see `comfyui/notes.md`); higher resolutions/batches not yet tested.

`gb10-writing` (`gemma4-26b-heretic-nvfp4`, 0.26) is sized to run **in
parallel** too: since 2026-10-10 qwen3.8-uc runs at 0.32 (291k tokens KV)
so qwen + bge-m3 + Gemma + ComfyUI all fit (0.64 of the 0.90 vLLM budget
plus ComfyUI). Use it for German stories / roleplay (see its notes for the
qwen / Small 4 / Gemma comparison; Small 4 was removed afterwards).

## Config layers

`common/box.env` (hardware) → `models/<slug>/model.env` (the model) →
a real environment variable at call time (`MAX_MODEL_LEN=32768 ./serve start`).
Each `.env` file uses `: "${K:=default}"`, so a set env var always wins.

`EXTRA_VLLM_ARGS` (a plain space-separated string) is appended verbatim to
`vllm serve` — the escape hatch for flags with no dedicated knob, e.g.
`EXTRA_VLLM_ARGS="--moe-backend marlin --attention-backend TRITON_ATTN" ./serve start`.
Empty by default; `gpt-oss-120b/model.env` documents the SM121 MXFP4 backend
overrides it exists for.

`RUNNER` (default `generate`) picks the vLLM 0.24.0 runner. `generate` gets the
full chat flag set (parsers, tool-choice, kv-cache-dtype, MTP, mm-limits).
`pooling` is for embedding models served at `/v1/embeddings` — a minimal flag
set with none of those; it also reads `DTYPE` (passed as `--dtype` when not
`auto`) and `ENFORCE_EAGER` (`1` adds `--enforce-eager`). CLS-pooling and
L2-norm come from the model's own config — see `bge-m3/`.

**Gotcha:** a few vars (`VLLM_IMAGE`, `HF_CACHE`, `VLLM_CACHE`, ...) already
get a value from `box.env`'s own `:=`, so a `model.env` line using `:=` on
one of *those* is a silent no-op (the var is already non-empty by the time
it runs) — a real per-model override needs a plain assignment instead, e.g.
`qwen3.8-27b-fp8/model.env`'s `VLLM_IMAGE` override (only replaces it if
still `box.env`'s stock default, so a one-off env var still wins).

---

## `llmctl` — interactive control

```bash
./llmctl
```

A one-screen terminal dashboard: every model, whether it's running, its pool
reservation (`GPU_MEM_UTIL` from `model.env`) and port. Type a number to
toggle it — running → stop, stopped → start. A start is refused up front if
the running models' `GPU_MEM_UTIL` sum plus the candidate would exceed the
pool budget (`POOL_BUDGET`, default `0.90` — host keeps ~0.10), telling you
which model to stop first. `(boot)` marks the models that autostart on reboot
(`qwen3.8-27b-fp8`, `bge-m3`); stopping one of those asks `y/N` first.
`q` or Ctrl-C quits.

It wraps `sudo systemctl start|stop llm-vllm@<slug>` when the systemd template
is installed (all models here), else `models/<slug>/serve` — same as doing it
by hand, minus the memory arithmetic. A model that was started as a bare
container (`serve start`) is still stopped correctly. The one-line role shown
per model comes from a `# llmctl: …` comment in each `model.env`.

`i` cycles the default image renderer of `gb10-rendering` through the
renderer registry (`qwen-image` → `flux` with the public defaults; shown in
its row). `p` cycles its default render
preset (`realistic` → `final` → `fast` → `anime`). Takes effect with the
next image, no restart — see the image-generation section below.

## Serve a model

The lower-level path (`llmctl` is the everyday one). Use `./serve` directly for
`status` / `logs`, or a one-off env override that systemd can't take, e.g.
`MAX_MODEL_LEN=32768 ./serve start`. `serve start`/`stop` defer to `systemctl`
while the model's systemd instance is active.

```bash
cd models/<slug>        # qwen3.8-27b-fp8 | bge-m3 | gpt-oss-120b | qwen3.8-27b-uc-nvfp4 | gemma4-26b-heretic-nvfp4

./serve                 # start: detached, wait for /health, print the API URL
./serve status          # systemd state + container health + API URLs + model list
./serve logs
./serve stop
```

Prerequisites: Docker + NVIDIA Container Toolkit; `.env` with `HF_TOKEN` (only
needed for gated repos); the weights downloaded —

No app-level auth: these vLLM endpoints run open (no `--api-key`). Access
control is the host firewall + IP whitelist instead — set up via
`common/setup-firewall.sh` (needs plain `sudo`; see the script header for
why a plain `ufw allow` alone doesn't work for these ports). The client IPs
are not in the repo: copy `common/firewall.local.env.example` to
`common/firewall.local.env` (git-ignored) and set them there. Access matrix:

| from | to this box | why |
|---|---|---|
| workstation (`WORKSTATION_IP`) | `22` (SSH/SCP), `8002` (ComfyUI), `8501` (DGX Dashboard) | operator access |
| harness server (`HARNESS_IP`) | `8000`/`8001`/`8002`/`8004` (LLM + render APIs) | the actual LLM clients (agent harness) |
| everyone else | nothing | default-deny |

`bash common/setup-firewall.sh --dry-run` prints the rules without touching
anything. To add a device/port, edit the rule block in the script and re-run
it (idempotent).

```bash
# qwen3.8-27b-fp8 (gated -> needs HF_TOKEN)
HF_HOME=/home/ss/models/hf-cache HF_TOKEN=hf_... \
  .hf-venv/bin/hf download orcarouter/Qwen3.8-27B-Uncensored-FP8

# gpt-oss-120b (public, no token; skip the Apple-metal and OpenAI-native
# weight formats -- vLLM only needs the HF-format safetensors shards)
HF_HOME=/home/ss/models/hf-cache \
  .hf-venv/bin/hf download openai/gpt-oss-120b --exclude "metal/*" --exclude "original/*"

# bge-m3 (public, no token)
HF_HOME=/home/ss/models/hf-cache .hf-venv/bin/hf download BAAI/bge-m3

```

First start is slow (weight load + torch.compile + CUDA-graph, + vision
encoder for qwen, + ~63 GiB cold load for gpt-oss-120b — `STARTUP_TIMEOUT` in each `model.env` is sized for this);
later starts are faster (compile cache persists). It prints:

```
  base URL (local)  : http://localhost:<port>/v1
  base URL (network): http://<lan-ip>:<port>/v1
  model id          : <served-name>
  auth              : none (open) -- firewall + IP whitelist is the access control
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-uncensored","messages":[{"role":"user","content":"hi"}]}'
```

- **No thinking:** add `"chat_template_kwargs": {"enable_thinking": false}`
  (qwen3.8-27b-fp8) — gpt-oss-120b's Harmony reasoning channel comes back in
  the `reasoning` field regardless, not toggleable per request the same way.
- **Vision / OCR (qwen3.8-27b-fp8 only):** `content` = a list of a `text`
  part and an `image_url` part (`data:image/png;base64,...`). gpt-oss-120b is
  text-only.
- **Tool calling:** both models run with `--enable-auto-tool-choice`; qwen
  uses its native `qwen3_coder` parser, gpt-oss-120b the Harmony
  `openai`/`openai_gptoss` parsers (see each `notes.md`).

**Embeddings (`bge-m3`, `RUNNER=pooling`):** served at `/v1/embeddings`, not
chat. 1024-dim dense vectors, CLS-pooled and L2-normalized (from the model
config — nothing to pass).

```bash
curl -s http://localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":["Hallo Welt","the quick brown fox"]}'
# -> data[0].embedding has length 1024
```

**Image generation (`comfyui/`, `gb10-rendering`, port 8002):** a different
engine entirely — ComfyUI, not vLLM, and not even raw ComfyUI on the wire.
**No-artifact requirement**: nothing (prompt text, image file) may survive
on this box once a generated image is handed off, so raw ComfyUI (which
would leave the output file and the prompt text in `/history` sitting
around) is bound to `127.0.0.1` only, never published. The only thing
published on 8002 is `comfyui/bridge.py`, a thin wrapper that submits the
workflow, waits for completion, reads the image bytes, **deletes the output
file and clears the ComfyUI history entry**, and returns the PNG bytes
directly in the response — one call, nothing left behind here afterward
regardless of what the caller does next:

```bash
curl -s http://localhost:8002/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a photorealistic portrait, golden hour lighting"}' \
  -o result.png
```

Renderers come from a registry, `comfyui/renderers.json`: by default `qwen`
(Qwen-Image-2512 + Lightning) and `flux` (FLUX.1-dev). A box can replace it
with its own renderers, workflows and LoRA files in the git-ignored
`comfyui/local/` (see `comfyui/notes.md` → "Renderer registry"). Without a
`"model"` field the box-wide default is used —
switched with `i` in `llmctl` (stored in
`~/models/comfyui-cache/bridge/default-model`); `"model": "flux"` in the
request overrides it. On a switch the bridge unloads the other model first.

Render settings: preset `realistic` (default) / `final` / `fast` / `anime`
(`comfyui/presets.json`, default switched with `p` in `llmctl`), or per
request `steps`, `guidance`, `cfg`, `sampler`, `scheduler`, `hires`,
`face_fix`, `hand_fix`, `negative` (switches to real CFG 3, ~1.5x time),
`lora_strength` (renderers with a LoRA node), `lora2` (the registry's
optional extra LoRA, off by default) — e.g. `{"prompt": "...", "preset": "final"}` or
`{"prompt": "...", "guidance": 3.0, "face_fix": false}`. Hires fix
(RealESRGAN) and face/hand fix (Impact-Pack FaceDetailer, ADetailer
equivalent) are built in. Details and how the values were chosen:
`comfyui/notes.md` → "Presets".

Renders time out after 1800 s (bridge, incl. queue wait; the job is aborted and cleaned up) — callers need a longer client timeout.

Intended caller: agents on the harness server (text prompt in, image bytes
out; the agent side posts the result to a chat channel and **must not persist
the bytes to disk on the harness server either** — that half of the
no-artifact requirement is the agent-side implementation's responsibility,
not enforceable from the LLM server).
See `comfyui/notes.md` for the full contract and caveats. Runs via
`comfyui/serve.sh`, not `common/serve.sh` — different engine, no
`GPU_MEM_UTIL` pool accounting, explicitly allowed to run in parallel with
the vLLM models (see that file's memory-risk section before assuming it
always fits).

## Persistent across reboots (systemd)

```bash
./serve install-service     # generates + enables  llm-vllm@<slug>  (sudo)

systemctl status llm-vllm@<slug>
journalctl -u  llm-vllm@<slug> -f
sudo systemctl restart llm-vllm@<slug>   # returns once serving again
./serve uninstall-service
```

The `llm-vllm@.service` template is installed, so **every** model runs as an
instance of it — `sudo systemctl start|stop llm-vllm@<slug>` (or `llmctl`)
for all of them. Enabled ones autostart on reboot:

| enabled (autostart) | systemd instance, no autostart |
|---|---|
| `qwen3.8-27b-uc-nvfp4`, `bge-m3`, `gemma4-26b-heretic-nvfp4` (+ `comfyui.service`) | `qwen3.8-27b-nvfp4`, `qwen3.8-27b-fp8`, `gpt-oss-120b` |

The on-demand ones come up only when you start them and stay down after a
reboot. To flip that: `sudo systemctl enable llm-vllm@<slug>` (autostart) /
`disable` (not) — but see the memory math below before enabling a big model.

One template unit, one instance per slug. systemd is the single supervisor —
`serve.sh ... run` execs `docker run --rm` in the foreground, no `--restart`
policy races it on boot. With the unit installed, use `systemctl` / `llmctl`,
not the bare `serve start`/`stop`.

**The template's `TimeoutStartSec` is 1200 s** (from the last `install-service`,
run by a `STARTUP_TIMEOUT=1200` model). That covers a warm start of any model;
a fully cold `gpt-oss-120b` (compile cache wiped, ~63 GiB load) could exceed
it — re-run `models/gpt-oss-120b/serve install-service` once to regenerate the
template at 1800 if that ever bites.

---

## Running multiple models

Each `model.env` sets its own `PORT` and `GPU_MEM_UTIL`. The 128 GB unified
pool is shared: **the sum of every running model's `GPU_MEM_UTIL` plus host
headroom must stay under 1.0.** e.g. two models at `0.40` each + `~0.15` host.
Container names are `vllm-<slug>`, benchmark containers `bench-<slug>`, so they
never collide.

**Fixed pair (systemd):** qwen (`0.65`) + bge-m3 (`0.12`) = `0.77`, ~0.23 of
the pool left for the host — fits, both boot-persistent.

`llmctl` enforces this as `sum(GPU_MEM_UTIL of running) + candidate <=
POOL_BUDGET` (0.90) before it starts anything.

**On-demand models — all need qwen3.8 stopped first (memory):**
- **`gpt-oss-120b` (`0.85`)** — ~66 GiB weights + ~34-35 GiB KV, ~101 GB of
  the ~109 GB carved.
Coding runs on `gb10-reasoning` (qwen3.8); the two `gb10-code` coders
(qwen3-coder-30b-a3b-fp8, deepseek-coder-v2-lite-fp8, port 8003) were removed
2026-10-11 — no gain over qwen3.8. Recipes + sweep results stay in git history
(last at `11e82c7`).

bge-m3 is small enough to leave running in every case. Ports are distinct per
model (8000/8001/8002/8004); container names are `vllm-<slug>`, benchmark
containers `bench-<slug>`, so nothing collides.

## Benchmarking

The harness reuses the model's serving port, so **stop the server first**.

```bash
models/<slug>/serve stop
sg docker -c ".hf-venv/bin/python common/bench/run.py --model models/<slug> --suite smoke"
sg docker -c ".hf-venv/bin/python common/bench/run.py --model models/<slug> --suite overnight"
python3 common/bench/report.py models/<slug>/bench/results/<suite>_<ts>
sg docker -c ".hf-venv/bin/python common/bench/ab_lmonly.py --model models/<slug>"
```

See `common/bench/README.md`. For `qwen3.8-27b-fp8` the report is at
`models/qwen3.8-27b-fp8/bench/results/overnight_20260827_191752/report.md`
(and <https://claude.ai/code/artifact/0843d0cb-f656-4f09-8e08-3a6fa499fd67>).

`gpt-oss-120b`'s `bench/matrix.yaml` sweep (backend/seqs/kv/context-scaling)
is done — results in `bench/results/*_20260831_*/`, winning config already
folded into `model.env`. The SM121 MXFP4 "null Harmony token" bug that
motivated the correctness-gating on `backend_*` (see `notes.md`) was
manually curl-verified **not** to reproduce on this box's image.

---

## Box-wide gotchas (`common/box.env`)

- **`VLLM_USE_DEEP_GEMM=0` is mandatory** on sm_121 — DeepGEMM's UE8M0 FP8 path
  asserts `Unknown recipe` during kernel warmup.
- **Unified memory:** `--gpu-memory-utilization` carves from the whole 128 GB
  pool. `nvidia-smi` reports `N/A` for GPU memory — use `free -m`.
- vLLM 0.24.0: MTP is `--speculative-config`, not `--num-speculative-tokens`.
- **NGC `26.07-py3` ships `xgrammar==0.2.0`**, but vLLM 0.24.0's tool-calling
  path needs `normalize_tool_choice` (added in xgrammar 0.2.4) — any request
  with `"tools"` set 500s (`cannot import name 'normalize_tool_choice'`) on
  the stock image. Any model that needs tool calling on this box needs the
  same `--no-deps` xgrammar bump; see
  `models/qwen3.8-27b-fp8/docker/Dockerfile.xgrammar024` for the pattern and
  its `model.env`'s `VLLM_IMAGE` override for how to wire it in.
- Models too big for one GB10 stay out (`Qwen3.8-Flash-Next`: `qwen4_exp`,
  ~180B MoE, ~186 GB in any quant, needs TP ≥ 8).

## License

Per-model — see each `models/<slug>/notes.md`. `qwen3.8-27b-fp8` is Apache 2.0,
**uncensored / abliterated** (no built-in guardrails; research use, add your own
moderation). `gpt-oss-120b` is Apache 2.0 (OpenAI, stock — keeps its own
safety training). `bge-m3` is MIT (BAAI). Scripts in this repo: MIT.
