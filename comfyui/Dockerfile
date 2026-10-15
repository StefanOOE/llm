# =============================================================================
#  comfyui/Dockerfile — ComfyUI for Qwen-Image-2512 text-to-image generation.
#
#  Base: nvcr.io/nvidia/pytorch, same 26.0x NGC release train as the vLLM
#  image (vllm-qwen:26.07-xgrammar024, built from nvcr.io/nvidia/vllm:26.07-py3)
#  -- confirmed via `docker manifest inspect` to ship an arm64 variant, which
#  this box (DGX Spark, aarch64) needs; most community ComfyUI images are
#  x86_64-only and won't run here at all. Using the matching NGC train means
#  the CUDA 13 / sm_121 (Blackwell, GB10) support already proven to work for
#  vLLM on this exact box should carry over, instead of gambling on a
#  different PyTorch build's Blackwell support.
#
#  ComfyUI pinned to v0.38.0 (latest at build time, confirmed via GitHub
#  release) -- ships native Qwen-Image support (added ~v0.37.0, Qwen-Image
#  2.1-specific optimizations in 0.38.0), no custom node pack needed.
# =============================================================================
FROM nvcr.io/nvidia/pytorch:26.07-py3

WORKDIR /opt/ComfyUI

RUN git clone --depth 1 --branch v0.38.0 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI

# requirements.txt lists `torch`/`torchvision`/`torchsde` unpinned -- pip
# leaves the NGC image's already-installed (CUDA 13/sm_121-matched) torch
# alone since no version constraint forces a reinstall. Do NOT add
# --upgrade here, that would pull a generic PyPI torch wheel without GB10
# support and break everything.
RUN pip install --no-cache-dir -r requirements.txt

# Face/hand fix (ADetailer equivalent): Impact Pack (FaceDetailer) + Impact
# Subpack (UltralyticsDetectorProvider, YOLO bbox detectors), pinned tags.
# Their deps (ultralytics especially) declare torch/torchvision/numpy and
# would happily replace the NGC builds with generic PyPI wheels without
# sm_121 support -- so pin exactly what's installed via a constraints file.
# ultralytics also drags in full opencv-python (needs libGL); swap it for
# the headless build. install.py is NOT run (it fetches SAM models etc.).
RUN git clone --depth 1 --branch 8.28 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git custom_nodes/ComfyUI-Impact-Pack \
 && git clone --depth 1 --branch 1.3.4 https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git custom_nodes/ComfyUI-Impact-Subpack \
 && python3 -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('torch','torchvision','numpy','triton','transformers','scipy','pillow')))" > /tmp/constraints.txt \
 && cat /tmp/constraints.txt \
 && pip install --no-cache-dir -c /tmp/constraints.txt \
      segment-anything scikit-image piexif opencv-python-headless GitPython dill "ultralytics>=8.3.162" \
 && pip uninstall -y opencv-python 2>/dev/null; \
    pip install --no-cache-dir -c /tmp/constraints.txt --force-reinstall --no-deps opencv-python-headless \
 && python3 -c "import torch, cv2, ultralytics; print('torch', torch.__version__, 'cv2', cv2.__version__, 'ultralytics', ultralytics.__version__)"

# ultralytics ships with usage telemetry ("sync") and online checks on --
# turn both off; detector models come from the bind-mounted models dir only.
ENV YOLO_OFFLINE=1 YOLO_CONFIG_DIR=/opt/ultralytics-config
RUN mkdir -p /opt/ultralytics-config && yolo settings sync=False >/dev/null && grep -q '"sync": false' /opt/ultralytics-config/Ultralytics/settings.json

# bridge.py is the ONLY thing this container exposes (port 8002) -- ComfyUI
# itself binds 127.0.0.1 only, launched as bridge.py's subprocess, so raw
# ComfyUI (/prompt, /view -- no cleanup guarantee) is never reachable from
# the LAN. See bridge.py's docstring for why (no-artifact-left-behind
# requirement).
COPY bridge.py presets.json renderers.json workflow-text2image.json workflow-flux-text2image.json /opt/ComfyUI/

EXPOSE 8002

# Models are bind-mounted from the host at /opt/ComfyUI/models (see
# comfyui/serve.sh) -- not baked into the image.
ENTRYPOINT ["python3", "/opt/ComfyUI/bridge.py"]
