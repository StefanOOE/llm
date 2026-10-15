#!/usr/bin/env bash
# =============================================================================
#  comfyui/serve.sh — launcher for the gb10-rendering ComfyUI container.
#  Not the shared vLLM common/serve.sh -- different engine entirely, see
#  comfyui/notes.md. Single instance, no per-model slug (unlike the LLM
#  identities there's only ever one rendering backend).
#
#     comfyui/serve.sh [start|run|stop|logs|status]
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
IMAGE="comfyui-qwen-image:v0.38.0"
CONTAINER="comfyui-gb10-rendering"
PORT=8002
# Host cache root (models, output, user, bridge); override per box via env.
COMFYUI_CACHE="${COMFYUI_CACHE:-/home/ss/models/comfyui-cache}"
MODELS_DIR="${COMFYUI_CACHE}/models"
OUTPUT_DIR="${COMFYUI_CACHE}/output"
USER_DIR="${COMFYUI_CACHE}/user"
# default-model / default-preset: box-wide defaults (renderer, render
# preset), written by llmctl, read by bridge.py per request
BRIDGE_DIR="${COMFYUI_CACHE}/bridge"

d() { docker "$@"; }

mkdir -p "$BRIDGE_DIR"
[ -f "$BRIDGE_DIR/default-model" ] || echo qwen > "$BRIDGE_DIR/default-model"
[ -f "$BRIDGE_DIR/default-preset" ] || echo realistic > "$BRIDGE_DIR/default-preset"

RUN_ARGS=(
  --name "$CONTAINER"
  --gpus all --ipc=host
  -p "${PORT}:8002"
  -v "${MODELS_DIR}:/opt/ComfyUI/models"
  -v "${OUTPUT_DIR}:/opt/ComfyUI/output"
  -v "${USER_DIR}:/opt/ComfyUI/user"
  -v "${BRIDGE_DIR}:/opt/bridge:ro"
)
# Box-local renderer registry + workflows (git-ignored comfyui/local/, see
# renderers.json): mounted only if present, else the public defaults apply.
LOCAL_DIR="${COMFYUI_LOCAL_DIR:-${HERE}/local}"
[ -f "${LOCAL_DIR}/renderers.json" ] && RUN_ARGS+=( -v "${LOCAL_DIR}:/opt/bridge-local:ro" )
RUN_ARGS+=(
  # generate timeout (s), bridge default 1800 -- one-off override:
  #   GENERATE_TIMEOUT_S=20 comfyui/serve.sh start
  -e "GENERATE_TIMEOUT_S=${GENERATE_TIMEOUT_S:-1800}"
  --health-cmd "curl -fsS -o /dev/null http://localhost:8002/health || exit 1"
  --health-interval=30s --health-timeout=5s --health-retries=3
  --health-start-period=180s
  "$IMAGE"
)

case "${1:-start}" in
  stop)
    d rm -f "$CONTAINER" >/dev/null 2>&1 && echo "stopped $CONTAINER" || echo "not running"
    ;;
  logs) d logs -f "$CONTAINER" ;;
  status)
    d ps --filter "name=^${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    curl -fsS -m 3 -o /dev/null -w "health: HTTP %{http_code}\n" "http://localhost:${PORT}/health" 2>&1 || echo "health: not responding"
    ;;
  run)
    # foreground: systemd's Restart=always supervises this directly.
    # NOTE: exec needs a real binary, not the d() shell function -- call
    # docker directly here (this bit bit us once already, see git log).
    d rm -f "$CONTAINER" >/dev/null 2>&1 || true
    exec docker run --rm "${RUN_ARGS[@]}"
    ;;
  start|*)
    d rm -f "$CONTAINER" >/dev/null 2>&1 || true
    d run -d --restart unless-stopped "${RUN_ARGS[@]}"
    echo "starting ${CONTAINER} -- http://localhost:${PORT}  (first load: weights + torch.compile, can take a few minutes)"
    ;;
esac
