#!/usr/bin/env bash
# =============================================================================
#  serve.sh  —  generic vLLM launcher for this box.
#
#     common/serve.sh <model-dir> [start|run|stop|logs|status|install-service|uninstall-service]
#
#  Config layers (later wins; a real env var beats both, for one-offs):
#     1. common/box.env            hardware facts (image, caches, sm_121 quirks)
#     2. models/<slug>/model.env   the model + its benchmark-tuned serving knobs
#     3. environment at call time  e.g.  MAX_MODEL_LEN=32768 ./serve start
#
#  Subcommands:
#     start            detached, wait for /health, print the API URL   (default)
#     run              foreground exec — used by the systemd unit
#     stop | logs | status
#     install-service  boot-persistent via  systemd  llm-vllm@<slug>   (sudo)
#     uninstall-service
#
#  Usually invoked through the per-model wrapper:  models/<slug>/serve start
# =============================================================================
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
LLM_ROOT="$(dirname "$COMMON_DIR")"

MODEL_DIR="${1:-}"
SUBCMD="${2:-start}"
if [ -z "$MODEL_DIR" ]; then
  echo "usage: $0 <model-dir> [start|run|stop|logs|status|install-service|uninstall-service]" >&2
  echo "models: $(cd "$LLM_ROOT/models" 2>/dev/null && echo */ | tr -d /)" >&2
  exit 2
fi
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
SLUG="$(basename "$MODEL_DIR")"
[ -f "$MODEL_DIR/model.env" ] || { echo "error: no model.env in $MODEL_DIR" >&2; exit 2; }

# -- layer 1 + 2 (:= inside the files means a real env var already wins) -- #
set -a
# shellcheck disable=SC1091
source "$COMMON_DIR/box.env"
# shellcheck disable=SC1090
source "$MODEL_DIR/model.env"
set +a

# -- box-wide secrets from LLM_ROOT/.env (model.env may already set API_KEY) - #
_env_get() { [ -f "$LLM_ROOT/.env" ] && sed -n "s/^$1=//p" "$LLM_ROOT/.env" | tail -1 | sed 's/^["'\'']//;s/["'\'']$//'; }
API_KEY="${API_KEY:-$(_env_get API_KEY)}"
HF_TOKEN="${HF_TOKEN:-$(_env_get HF_TOKEN)}"; export HF_TOKEN

# -- derived defaults -------------------------------------------------- #
CONTAINER="${CONTAINER:-vllm-$SLUG}"
SERVED_NAME="${SERVED_NAME:-$SLUG}"
HOST="${HOST:-0.0.0.0}"
UNIT="llm-vllm@${SLUG}.service"
VISION_DESC=""            # filled by assemble_args
# how the user invoked us, for hint text
if [ -x "$MODEL_DIR/serve" ]; then INVOKED_AS="models/${SLUG}/serve"
else INVOKED_AS="common/serve.sh models/${SLUG}"; fi

# shellcheck disable=SC1091
source "$COMMON_DIR/lib.sh"
main "$SUBCMD"
