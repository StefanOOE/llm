#!/usr/bin/env python3
"""bridge.py -- the ONLY thing this container exposes to the LAN (port 8002).

Why this exists: the user requires that NO artifact (prompt text, image
file) survives on this box once a generated image has been handed off --
raw ComfyUI's own API (/prompt + /view) has no such guarantee: the output
file and the prompt text in /history just sit there afterward. So ComfyUI
itself binds to 127.0.0.1 only (never published to the host/LAN -- see
serve.sh), and this bridge is the single externally-reachable entry point:

  POST /generate   {"prompt": "...", "width": 1024, "height": 1024, "seed": null,
                    "model": "qwen"}
    model: a renderer id from the renderer registry -- renderers.json next
           to this file ("qwen" = Qwen-Image-2512 + Lightning, "flux" =
           FLUX.1-dev), or, if present, the box-local registry mounted at
           /opt/bridge-local/renderers.json (git-ignored comfyui/local/),
           which replaces it and may add renderers/workflows of its own.
           Optional -- if omitted, the box-wide default renderer is used:
           /opt/bridge/default-model (host: comfyui-cache/bridge/, written
           by llmctl's renderer toggle), re-read on every request, so
           switching needs no restart. Falls back to the registry default.
    Render settings (all optional; request field > preset > nothing):
      preset    "realistic" (default) | "final" | "fast" | "anime" -- see
                presets.json; box-wide default in /opt/bridge/default-preset
                (llmctl `p`), re-read per request.
      steps, cfg, guidance, sampler, scheduler   (flux stack only; qwen's
                Lightning is fixed to 4 steps / cfg 1 / euler and ignores them)
      negative  "..." negative prompt (flux stack). Needs real CFG > 1: without
                an explicit "cfg" it switches to cfg 3.0 (~2x render time);
                qwen ignores it. Like the prompt, never logged.
      lora_strength  0-2, strength of the renderer's own LoRA (renderers with
                a "lora_node" in the registry only, default 1.0)
      lora2     true|false|0-2  optional extra LoRA ("extra_lora" in the
                registry, only on the stacks listed there), default off;
                true = strength 1.0
      hires     true|false|{"scale": 1-2, "denoise", "steps"}  RealESRGAN 4x
                -> scale, second sampling pass
      face_fix, hand_fix   true|false|{"steps", "denoise"}  Impact-Pack
                FaceDetailer with YOLO face/hand detectors (ADetailer equivalent)
    Invalid values -> HTTP 400. The resolved settings (never the prompt) are
    logged and returned in the X-Render-Settings response header.
    When the model differs from the previous request's, all models are
    unloaded first (ComfyUI POST /free) -- otherwise ComfyUI keeps the old
    stack resident and both (~48 GiB) sit in the unified pool next to vLLM.
    -> submits the model's workflow (prompt/seed/size substituted) to
       ComfyUI, polls for completion, reads the resulting PNG bytes,
       DELETES the output file and clears ComfyUI's /history entry for that
       prompt_id, then returns the PNG bytes directly in the response body
       (image/png). No separate fetch step for the caller (the agent harness) --
       nothing to leave lying around on either side once this response is
       sent and the caller discards the bytes after posting to its chat channel
       (that half of the guarantee is the caller's responsibility, not
       enforceable from here -- see comfyui/notes.md).
  GET  /health     -> proxies ComfyUI's own /system_stats.

Cleanup happens unconditionally once the bytes are read, regardless of
whether the caller's later chat post succeeds -- this box has no way to
know that outcome and shouldn't hold a file hostage waiting for a
confirmation callback that might never come.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

from aiohttp import web, ClientSession, ClientTimeout

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"
BRIDGE_PORT = int(os.environ.get("BRIDGE_PORT", "8002"))
OUTPUT_DIR = Path("/opt/ComfyUI/output")
HERE = Path(__file__).resolve().parent
# Renderer registry: the box-local one (git-ignored comfyui/local/, mounted
# read-only) replaces the public default entirely. Workflow paths are
# relative to the registry file. See renderers.json for the format.
LOCAL_REGISTRY = Path(os.environ.get("BRIDGE_LOCAL_REGISTRY", "/opt/bridge-local/renderers.json"))
REGISTRY_FILE = LOCAL_REGISTRY if LOCAL_REGISTRY.is_file() else HERE / "renderers.json"
REGISTRY = json.loads(REGISTRY_FILE.read_text())
RENDERERS = REGISTRY["renderers"]
WORKFLOWS = {k: REGISTRY_FILE.parent / v["workflow"] for k, v in RENDERERS.items()}
# which loaded weights each model needs -- switching within a stack (e.g. a
# base model and the same base with a LoRA patched on) must not trigger an unload
STACK = {k: v["stack"] for k, v in RENDERERS.items()}
LORA_NODE = {k: v["lora_node"] for k, v in RENDERERS.items() if v.get("lora_node")}
EXTRA_LORA = REGISTRY.get("extra_lora") or None   # {"file": ..., "stacks": [...]}
DEFAULT_MODEL = REGISTRY.get("default", next(iter(RENDERERS)))
DEFAULT_MODEL_FILE = Path("/opt/bridge/default-model")
DEFAULT_PRESET_FILE = Path("/opt/bridge/default-preset")
PRESETS = json.loads((HERE / "presets.json").read_text())
FREE_SETTLE_S = 3  # /free only sets a flag the prompt worker handles between queue items
# Generous: counted from submission, so it includes queue wait behind other
# requests; a model switch means unloading the other stack and loading
# ~12-30 GiB of weights; preset "final" adds a hires pass + detailers; and
# everything slows down when vLLM is busy on the same GPU. Env-overridable
# (serve.sh passes GENERATE_TIMEOUT_S through). Clients need a longer one.
GENERATE_TIMEOUT_S = int(os.environ.get("GENERATE_TIMEOUT_S", "1800"))
ABORT_WAIT_S = 120      # after cancel/interrupt: how long to wait for ComfyUI to let go of the job
TEMP_DIR = Path("/opt/ComfyUI/temp")

PROMPT_NODE_ID = "6"
SEED_NODE_ID = "3"      # main KSampler
LATENT_NODE_ID = "5"
DECODE_NODE_ID = "8"
GUIDANCE_NODE_ID = "14"  # FluxGuidance, flux workflows only
SAVE_NODE_ID = "13"

UPSCALE_MODEL = "RealESRGAN_x4plus.pth"          # models/upscale_models/
FACE_DETECTOR = "bbox/face_yolov8m.pt"           # models/ultralytics/
HAND_DETECTOR = "bbox/hand_yolov8s.pt"
LORA2_NODE_ID = "40"    # optional extra LoRA (registry "extra_lora"), spliced in by build_graph()

NEGATIVE_NODE_ID = "7"
NEG_GUIDANCE_NODE_ID = "15"   # FluxGuidance on the negative branch, added with a negative prompt
NEGATIVE_DEFAULT_CFG = 3.0    # real CFG used when a negative prompt is given without "cfg"
LIMITS = {"steps": (1, 80), "cfg": (1.0, 15.0), "guidance": (0.0, 15.0), "lora_strength": (0.0, 2.0)}
FIX_LIMITS = {"steps": (1, 60), "denoise": (0.05, 1.0), "scale": (1.0, 2.0)}


class BadRequest(ValueError):
    pass


def _read_choice(path: Path, valid, fallback: str) -> str:
    try:
        v = path.read_text().strip()
    except OSError:
        return fallback
    return v if v in valid else fallback


def default_model() -> str:
    return _read_choice(DEFAULT_MODEL_FILE, WORKFLOWS, DEFAULT_MODEL)


def default_preset() -> str:
    return _read_choice(DEFAULT_PRESET_FILE, PRESETS["presets"], PRESETS["default"])


def _clamp(name: str, v, lo, hi, cast):
    try:
        v = cast(v)
    except (TypeError, ValueError):
        raise BadRequest(f"'{name}' must be a number")
    if not lo <= v <= hi:
        raise BadRequest(f"'{name}' must be between {lo} and {hi}")
    return v


def resolve_settings(model: str, body: dict, samplers: dict) -> dict:
    """request fields > preset > nothing. Returns a fully-populated dict.
    Never contains prompt text -- it's logged and sent back as a header."""
    preset = body.get("preset") or default_preset()
    if preset not in PRESETS["presets"]:
        raise BadRequest(f"'preset' must be one of {sorted(PRESETS['presets'])}")
    stack = STACK[model]
    st = json.loads(json.dumps(PRESETS["presets"][preset][stack]))  # deep copy
    st["preset"] = preset

    if stack == "qwen":
        # Lightning LoRA is distilled to 4 steps / cfg 1 / euler -- overriding
        # those only produces broken images, so they're ignored on purpose.
        ignored = [k for k in ("steps", "cfg", "guidance", "sampler", "scheduler") if k in body]
        if ignored:
            print(f"[bridge] qwen: ignoring {ignored} (Lightning is fixed to its preset values)", flush=True)
    else:
        for k in ("steps", "cfg", "guidance"):
            if k in body:
                lo, hi = LIMITS[k]
                st[k] = _clamp(k, body[k], lo, hi, int if k == "steps" else float)
        for k, valid in (("sampler", samplers["sampler"]), ("scheduler", samplers["scheduler"])):
            if k in body:
                if body[k] not in valid:
                    raise BadRequest(f"'{k}' must be one of {valid}")
                st[k] = body[k]

    # Negative prompt: only does anything with real CFG > 1 (FLUX-dev is
    # distilled to cfg 1, where the negative branch isn't evaluated at all),
    # so a negative without an explicit "cfg" switches to NEGATIVE_DEFAULT_CFG
    # -- roughly doubles render time. Only a flag goes into st, not the text.
    neg = body.get("negative")
    if neg is not None and not isinstance(neg, str):
        raise BadRequest("'negative' must be a string")
    st["negative"] = False
    if neg and neg.strip():
        if stack == "qwen":
            print("[bridge] qwen: ignoring negative (Lightning is fixed to cfg 1)", flush=True)
        else:
            st["negative"] = True
            if "cfg" not in body:
                st["cfg"] = NEGATIVE_DEFAULT_CFG
            elif st["cfg"] <= 1.0:
                print("[bridge] negative given with cfg 1 -- it has no effect", flush=True)

    # LoRAs. lora_strength: the LoRA node a renderer's workflow always loads
    # (registry "lora_node"). lora2: the optional extra LoRA (registry
    # "extra_lora"), only on the stacks listed there, off unless requested --
    # true = strength 1.0, a number = that strength, false/0 = off.
    if model in LORA_NODE:
        st["lora_strength"] = 1.0
        if "lora_strength" in body:
            st["lora_strength"] = _clamp("lora_strength", body["lora_strength"], *LIMITS["lora_strength"], float)
    elif "lora_strength" in body:
        print(f"[bridge] {model}: ignoring lora_strength (no LoRA node in this workflow)", flush=True)
    if EXTRA_LORA and stack in EXTRA_LORA.get("stacks", []):
        v = body.get("lora2", False)
        if isinstance(v, bool):
            st["lora2"] = 1.0 if v else 0.0
        else:
            st["lora2"] = _clamp("lora2", v, *LIMITS["lora_strength"], float)
    elif "lora2" in body:
        print(f"[bridge] {model}: ignoring lora2 (no extra LoRA configured for this stack)", flush=True)

    for stage in ("hires", "face_fix", "hand_fix"):
        if stage not in body:
            continue
        v = body[stage]
        if isinstance(v, bool):
            st[stage]["enabled"] = v
        elif isinstance(v, dict):
            st[stage]["enabled"] = bool(v.get("enabled", True))
            for k in ("steps", "denoise", "scale"):
                if k in v and k in st[stage]:
                    lo, hi = FIX_LIMITS[k]
                    st[stage][k] = _clamp(f"{stage}.{k}", v[k], lo, hi, int if k == "steps" else float)
        else:
            raise BadRequest(f"'{stage}' must be true/false or an object")
    return st


def load_workflow_template(model: str) -> dict:
    return json.loads(WORKFLOWS[model].read_text())


def _detailer(image, detector_node: str, ks: dict, fix: dict, seed: int, guide_size: int, crop_factor: float) -> dict:
    return {"class_type": "FaceDetailer", "inputs": {
        "image": image, "model": ks["model"], "clip": ["9", 0], "vae": ["10", 0],
        "guide_size": guide_size, "guide_size_for": True, "max_size": 1024,
        "seed": seed, "steps": fix["steps"], "cfg": ks["cfg"],
        "sampler_name": ks["sampler_name"], "scheduler": ks["scheduler"],
        "positive": ks["positive"], "negative": ks["negative"],
        "denoise": fix["denoise"], "feather": 5, "noise_mask": True, "force_inpaint": True,
        "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": crop_factor,
        "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
        "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
        "sam_mask_hint_use_negative": "False", "drop_size": 10,
        "bbox_detector": [detector_node, 0], "wildcard": "", "cycle": 1,
        "inpaint_model": False, "noise_mask_feather": 20,
    }}


def build_graph(model: str, prompt: str, width: int, height: int, seed: int, st: dict, negative: str = "") -> dict:
    graph = load_workflow_template(model)
    graph.pop("_comment", None)
    graph[PROMPT_NODE_ID]["inputs"]["text"] = prompt
    graph[LATENT_NODE_ID]["inputs"]["width"] = width
    graph[LATENT_NODE_ID]["inputs"]["height"] = height
    ks = graph[SEED_NODE_ID]["inputs"]
    ks.update(seed=seed, steps=st["steps"], cfg=st["cfg"],
              sampler_name=st["sampler"], scheduler=st["scheduler"])
    if GUIDANCE_NODE_ID in graph and "guidance" in st:
        graph[GUIDANCE_NODE_ID]["inputs"]["guidance"] = st["guidance"]
    if st.get("negative"):
        graph[NEGATIVE_NODE_ID]["inputs"]["text"] = negative
        if GUIDANCE_NODE_ID in graph:
            # same distilled-guidance embedding on both branches, so CFG
            # contrasts only the text, not two different guidance values
            graph[NEG_GUIDANCE_NODE_ID] = {"class_type": "FluxGuidance", "inputs": {
                "guidance": st["guidance"], "conditioning": [NEGATIVE_NODE_ID, 0]}}
            ks["negative"] = [NEG_GUIDANCE_NODE_ID, 0]
    if model in LORA_NODE and "lora_strength" in st:
        graph[LORA_NODE[model]]["inputs"]["strength_model"] = st["lora_strength"]
    if st.get("lora2"):
        # chained after whatever feeds the sampler (UNET or the renderer's own
        # LoRA); hires/detailers below copy ks["model"], so they get it too
        graph[LORA2_NODE_ID] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ks["model"], "lora_name": EXTRA_LORA["file"], "strength_model": st["lora2"]}}
        ks["model"] = [LORA2_NODE_ID, 0]

    # Optional post stages, chained on the decoded image. Everything stays in
    # memory -- only the final SaveImage (13) writes a file (no-artifact rule).
    image = [DECODE_NODE_ID, 0]
    vae = ["10", 0]
    h = st["hires"]
    if h["enabled"]:
        graph["20"] = {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}}
        graph["21"] = {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["20", 0], "image": image}}
        graph["22"] = {"class_type": "ImageScaleBy", "inputs": {"image": ["21", 0], "upscale_method": "lanczos", "scale_by": h["scale"] / 4}}
        graph["23"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["22", 0], "vae": vae}}
        graph["24"] = {"class_type": "KSampler", "inputs": {**ks, "steps": h["steps"], "denoise": h["denoise"], "latent_image": ["23", 0]}}
        graph["25"] = {"class_type": "VAEDecode", "inputs": {"samples": ["24", 0], "vae": vae}}
        image = ["25", 0]
    guide = 768 if STACK[model] == "flux" else 512
    if st["face_fix"]["enabled"]:
        graph["30"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": FACE_DETECTOR}}
        graph["31"] = _detailer(image, "30", ks, st["face_fix"], seed, guide, 3.0)
        image = ["31", 0]
    if st["hand_fix"]["enabled"]:
        graph["32"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": HAND_DETECTOR}}
        graph["33"] = _detailer(image, "32", ks, st["hand_fix"], seed, guide, 2.5)
        image = ["33", 0]
    graph[SAVE_NODE_ID]["inputs"]["images"] = image
    # Unique per request: ComfyUI skips nodes whose inputs are unchanged, so an
    # identical repeat request (same prompt + seed) would otherwise not re-run
    # SaveImage and report the previous file -- which we already deleted.
    # This way only SaveImage re-runs (upstream comes from the in-memory cache).
    graph[SAVE_NODE_ID]["inputs"]["filename_prefix"] = f"gb10-rendering_{uuid.uuid4().hex[:12]}"
    return graph


async def wait_comfy_healthy(session: ClientSession, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            async with session.get(f"{COMFY_BASE}/system_stats", timeout=ClientTimeout(total=3)) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        await asyncio.sleep(2)
    raise RuntimeError("ComfyUI did not become healthy in time")


async def unload_if_switching(app: web.Application, model: str) -> None:
    last = app["last_model"]
    app["last_model"] = model
    if last is None or STACK[last] == STACK[model]:
        return
    print(f"[bridge] renderer switch {last} -> {model}: unloading models", flush=True)
    async with app["session"].post(f"{COMFY_BASE}/free", json={"unload_models": True, "free_memory": True}):
        pass
    await asyncio.sleep(FREE_SETTLE_S)


async def _job_in_queue(session: ClientSession, prompt_id: str) -> tuple[bool, bool]:
    """(running, pending) for prompt_id in ComfyUI's queue."""
    async with session.get(f"{COMFY_BASE}/queue") as r:
        q = await r.json()
    running = any(item[1] == prompt_id for item in q.get("queue_running", []))
    pending = any(item[1] == prompt_id for item in q.get("queue_pending", []))
    return running, pending


async def abort_job(session: ClientSession, prompt_id: str) -> None:
    """Stop a job we gave up on (timeout, error, cancellation) so it can't
    finish later and leave its image + prompt behind with nobody to clean up.
    Pending -> removed from the queue; running -> /interrupt (only if it's
    really ours -- /interrupt hits whatever is running). Then wait until
    ComfyUI has let go of it, so the file/history sweep after this is final."""
    try:
        async with session.post(f"{COMFY_BASE}/queue", json={"delete": [prompt_id]}):
            pass
        deadline = time.time() + ABORT_WAIT_S
        while time.time() < deadline:
            running, pending = await _job_in_queue(session, prompt_id)
            if not running and not pending:
                return
            if running:
                async with session.post(f"{COMFY_BASE}/interrupt"):
                    pass
            await asyncio.sleep(1)
        print(f"[bridge] WARNING: job {prompt_id} still in ComfyUI after {ABORT_WAIT_S}s abort wait", flush=True)
    except Exception as e:
        print(f"[bridge] WARNING: aborting job {prompt_id} failed: {e}", flush=True)


async def cleanup_job(session: ClientSession, prompt_id: str, prefix: str) -> None:
    """Delete every output file of this request (unique filename prefix) and
    its /history entry (which holds the prompt text)."""
    for f in OUTPUT_DIR.rglob(f"{prefix}*"):
        f.unlink(missing_ok=True)
    try:
        async with session.post(f"{COMFY_BASE}/history", json={"delete": [prompt_id]}):
            pass
    except Exception as e:
        print(f"[bridge] WARNING: clearing history for {prompt_id} failed: {e}", flush=True)


async def generate(session: ClientSession, model: str, prompt: str, width: int, height: int, seed: int, st: dict, negative: str = "") -> bytes:
    client_id = str(uuid.uuid4())
    graph = build_graph(model, prompt, width, height, seed, st, negative)
    prefix = graph[SAVE_NODE_ID]["inputs"]["filename_prefix"]

    async with session.post(f"{COMFY_BASE}/prompt", json={"prompt": graph, "client_id": client_id}) as r:
        if r.status != 200:
            raise RuntimeError(f"/prompt failed: {r.status} {await r.text()}")
        data = await r.json()
    prompt_id = data["prompt_id"]

    finished = False
    try:
        deadline = time.time() + GENERATE_TIMEOUT_S
        history = None
        while time.time() < deadline:
            async with session.get(f"{COMFY_BASE}/history/{prompt_id}") as r:
                h = await r.json()
                if prompt_id in h:
                    history = h[prompt_id]
                    break
            await asyncio.sleep(1)
        if history is None:
            raise RuntimeError(f"generation timed out after {GENERATE_TIMEOUT_S}s")
        finished = True

        status = history.get("status", {})
        if status.get("status_str") == "error":
            raise RuntimeError(f"ComfyUI reported an error: {status}")

        images = history.get("outputs", {}).get("13", {}).get("images", [])
        if not images:
            raise RuntimeError("no image in ComfyUI output (unexpected graph shape -- see notes.md)")
        img = images[0]
        return (OUTPUT_DIR / img.get("subfolder", "") / img["filename"]).read_bytes()
    finally:
        # Runs on success, error, timeout and cancellation alike -- no
        # artifact survives either way. A job we stopped waiting for is
        # aborted first, otherwise it would write its file after the sweep.
        if not finished:
            await abort_job(session, prompt_id)
        await cleanup_job(session, prompt_id, prefix)


async def handle_generate(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    prompt = body.get("prompt")
    if not prompt or not isinstance(prompt, str):
        return web.json_response({"error": "'prompt' (string) is required"}, status=400)
    width = int(body.get("width", 1024))
    height = int(body.get("height", 1024))
    seed = body.get("seed")
    seed = random.randint(0, 2**32 - 1) if seed is None else int(seed)
    model = body.get("model") or default_model()
    if model not in WORKFLOWS:
        return web.json_response({"error": f"'model' must be one of {sorted(WORKFLOWS)}"}, status=400)
    try:
        st = resolve_settings(model, body, request.app["samplers"])
    except BadRequest as e:
        return web.json_response({"error": str(e)}, status=400)
    # settings only, never the prompt (no-artifact rule also covers logs)
    summary = {"model": model, "seed": seed, "size": f"{width}x{height}", **st}
    print(f"[bridge] render {json.dumps(summary, separators=(',', ':'))}", flush=True)

    session: ClientSession = request.app["session"]
    try:
        await unload_if_switching(request.app, model)
        image_bytes = await generate(session, model, prompt, width, height, seed, st, body.get("negative") or "")
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)
    return web.Response(body=image_bytes, content_type="image/png",
                        headers={"X-Render-Settings": json.dumps(summary, separators=(",", ":"))})


async def handle_health(request: web.Request) -> web.Response:
    session: ClientSession = request.app["session"]
    try:
        async with session.get(f"{COMFY_BASE}/system_stats", timeout=ClientTimeout(total=3)) as r:
            return web.json_response({"comfyui": "up"}, status=200 if r.status == 200 else 503)
    except Exception:
        return web.json_response({"comfyui": "down"}, status=503)


def sweep_leftovers() -> int:
    """Safety net after a crash/kill mid-job: nothing in output/ or temp/
    is ever meant to outlive a request, so empty both at startup."""
    n = 0
    for d in (OUTPUT_DIR, TEMP_DIR):
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file() or f.is_symlink():
                f.unlink(missing_ok=True)
                n += 1
    return n


async def on_startup(app: web.Application) -> None:
    app["session"] = ClientSession()
    app["last_model"] = None
    await wait_comfy_healthy(app["session"])
    n = sweep_leftovers()
    async with app["session"].post(f"{COMFY_BASE}/history", json={"clear": True}):
        pass
    print(f"[bridge] startup sweep: removed {n} leftover file(s), history cleared; generate timeout {GENERATE_TIMEOUT_S}s", flush=True)
    print(f"[bridge] renderer registry {REGISTRY_FILE}: {sorted(RENDERERS)}, default {DEFAULT_MODEL}", flush=True)
    async with app["session"].get(f"{COMFY_BASE}/object_info/KSampler") as r:
        req = (await r.json())["KSampler"]["input"]["required"]
    app["samplers"] = {"sampler": req["sampler_name"][0], "scheduler": req["scheduler"][0]}
    print("[bridge] ComfyUI is healthy, bridge ready", flush=True)


async def on_cleanup(app: web.Application) -> None:
    await app["session"].close()


def main() -> None:
    # ComfyUI itself: localhost-only, never published to the host (see
    # serve.sh -- only BRIDGE_PORT is -p'd). Supervised as a child process;
    # if it dies, we die too (systemd's Restart=always on comfyui.service
    # then restarts the whole container).
    comfy = subprocess.Popen(
        # --disable-metadata: SaveImage otherwise embeds the whole graph --
        # prompt and negative text included -- as PNG tEXt chunks, so the
        # prompt would travel inside every image (chat-app document uploads keep it).
        [sys.executable, "main.py", "--listen", COMFY_HOST, "--port", str(COMFY_PORT), "--disable-metadata"],
        cwd="/opt/ComfyUI",
    )

    app = web.Application(client_max_size=1024**2)
    app.router.add_post("/generate", handle_generate)
    app.router.add_get("/health", handle_health)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    try:
        web.run_app(app, host="0.0.0.0", port=BRIDGE_PORT, print=None)
    finally:
        comfy.terminate()
        comfy.wait(timeout=30)


if __name__ == "__main__":
    main()
