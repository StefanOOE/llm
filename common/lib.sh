# =============================================================================
#  lib.sh  —  shared implementation for common/serve.sh
#  Sourced AFTER box.env + model.env + the derived vars in serve.sh, so every
#  function below just reads the globals: MODEL SERVED_NAME CONTAINER HOST PORT
#  MAX_MODEL_LEN MAX_NUM_SEQS GPU_MEM_UTIL KV_CACHE_DTYPE MTP_TOKENS
#  LANGUAGE_MODEL_ONLY MM_* REASONING_PARSER TOOL_CALL_PARSER EXTRA_VLLM_ARGS
#  VLLM_IMAGE HF_CACHE VLLM_CACHE VLLM_USE_DEEP_GEMM STARTUP_TIMEOUT
#  HF_TOKEN  SLUG MODEL_DIR LLM_ROOT COMMON_DIR UNIT BOX_*
#
#  No vLLM --api-key / auth: these endpoints are open HTTP, access control is
#  the host firewall + IP whitelist, not an application-level key.
#
#  EXTRA_VLLM_ARGS (optional, default ""): a plain space-separated string of
#  extra flags for `vllm serve`, word-split as-is (no quoting/JSON values) —
#  e.g. a model whose optimal --moe-backend / --attention-backend was found
#  by benchmarking, not read from the checkpoint. Unset -> no-op for models
#  (like qwen3.8-27b-fp8) that don't need it.
#
#  DOCKER_ENV (optional, default ""): space-separated KEY=VALUE tokens, each
#  passed to `docker run` as `-e KEY=VALUE` (container env, NOT a vLLM flag).
#  For vLLM/Triton/torch env vars that have no serve flag -- e.g.
#  VLLM_TRITON_FORCE_FIRST_CONFIG=1 (skip the OOM-prone GDN autotuner on
#  sm_121, see qwen3.8-27b-fp8/model.env), or CUDA_LAUNCH_BLOCKING=1 for a
#  diagnostic run. Word-split as-is; no spaces inside a value. Unset -> no
#  extra -e flags, identical to before this var existed.
#
#  RUNNER (optional, default "generate"): vLLM 0.24.0's `--runner`.
#    generate  chat/completions model — the full flag set below
#              (parsers, tool-choice, kv-cache-dtype, MTP, mm-limits).
#    pooling   embedding model served at /v1/embeddings — a deliberately
#              minimal flag set, none of the generate-only flags. Reads
#              DTYPE and ENFORCE_EAGER; ignores MTP_TOKENS / KV_CACHE_DTYPE /
#              MAX_NUM_SEQS / REASONING_PARSER / TOOL_CALL_PARSER / MM_* /
#              LANGUAGE_MODEL_ONLY (a pooling model.env need not set them).
#              CLS-pooling + L2-norm come from the model's own config
#              (1_Pooling/, sentence_bert_config.json) — do not hand-set
#              --pooler-config unless a test shows it's wrong.
#  DTYPE (optional): passed as `--dtype` only when set and != "auto".
#  ENFORCE_EAGER (optional, default 0): 1 adds `--enforce-eager`.
#
#  VLLM_MODEL_PATH (optional): if set, passed to `vllm serve` INSTEAD of
#  $MODEL (which still names the repo id everywhere else -- preflight's
#  weights-present check, the banner, error messages). Needed when the local
#  HF cache is a deliberately partial snapshot (e.g. gpt-oss-120b's --exclude
#  metal/original): vLLM 0.24's get_model_path() calls HF's snapshot_download
#  even with HF_HUB_OFFLINE=1, which raises IncompleteSnapshotError for files
#  missing on purpose. Its actual first line is `if os.path.exists(model):
#  return model` -- a local path (container-side, under /hf-cache/...) skips
#  the whole hub check. See gpt-oss-120b/model.env for how it's resolved.
#
#  PREFIX_CACHING (optional, generate runner only): "1" adds
#  --enable-prefix-caching, "0" adds --no-enable-prefix-caching, unset ->
#  no flag at all (vLLM's own internal default, unchanged from before this
#  var existed). Mirrors common/bench/run.py's own prefix_caching handling,
#  which always sets one or the other explicitly per sweep config -- this
#  var adds the "don't touch it" third state serving needs but the bench
#  harness doesn't. See qwen3.8-27b-fp8/model.env for a benchmarked example
#  (prefix_cache_ab suite: -25/-27% TTFT, +22% throughput under concurrency).
#
#  REASONING_PARSER (optional, generate runner only): passed as
#  --reasoning-parser when non-empty; unset/empty -> flag omitted entirely.
#  For a non-thinking model (e.g. a *-Coder-*-Instruct checkpoint with no
#  reasoning channel) leave it unset rather than pointing it at some other
#  model's parser -- there's no reasoning_start/end token for it to find.
#
#  TOOL_CALL_PARSER (optional, generate runner only): when non-empty, adds
#  BOTH --enable-auto-tool-choice and --tool-call-parser (they're a pair --
#  auto-tool-choice with no parser, or a parser with tool-choice off, are
#  both meaningless). Unset/empty -> neither flag, i.e. tool calling stays
#  off. Use this for a model with no vLLM-supported tool-call format on this
#  box rather than guessing a parser and letting `tools` requests 500.
#
#  TRUST_REMOTE_CODE (optional, generate runner only, default "1"): "1" adds
#  --trust-remote-code (the long-standing default), "0" omits it. Set to "0"
#  for a checkpoint whose config.json carries an `auto_map` pointing at a
#  DIFFERENT repo's custom modeling code (e.g. a re-quant of a base model):
#  with HF_HUB_OFFLINE=1 that code isn't cached and AutoConfig.from_pretrained
#  hard-fails trying to fetch it, even though vLLM has a NATIVE class for the
#  architecture and doesn't need the remote code at all. See
#  deepseek-coder-v2-lite-fp8/model.env (removed 2026-10-11; git history, 11e82c7).
# =============================================================================

# -- docker (works with or without docker-group membership) --------------- #
if docker info >/dev/null 2>&1; then
  DOCKER_DIRECT=1
  d() { docker "$@"; }
  exec_docker() { exec docker "$@"; }
elif sg docker -c 'docker info' >/dev/null 2>&1; then
  DOCKER_DIRECT=0
  d() { sg docker -c "docker $(printf '%q ' "$@")"; }
  exec_docker() { exec sg docker -c "docker $(printf '%q ' "$@")"; }
else
  echo "error: cannot reach the Docker daemon (not in the 'docker' group?)." >&2
  echo "       sudo usermod -aG docker \$USER   # then open a new login shell" >&2
  exit 1
fi

lan_ip() {
  local ip
  ip="$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.*src \([0-9.]*\).*/\1/p' | head -1)"
  echo "${ip:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
}

# -- assemble the two arg arrays into globals RUN_COMMON / VLLM_ARGS ------ #
assemble_args() {
  RUNNER="${RUNNER:-generate}"

  # container-level flags — identical for every runner
  HEALTH_ARGS=(
    --health-cmd "curl -fsS -o /dev/null http://localhost:${PORT}/health || exit 1"
    --health-interval=30s --health-timeout=5s --health-retries=3
    --health-start-period=600s
  )

  # generic container-env passthrough: DOCKER_ENV="FOO=1 BAR=2" -> -e FOO=1 -e BAR=2
  local docker_env=()
  # shellcheck disable=SC2086  # intentional word-split of a flat KEY=VAL list
  for _kv in ${DOCKER_ENV:-}; do docker_env+=( -e "$_kv" ); done

  RUN_COMMON=(
    --name "$CONTAINER"
    --gpus all --ipc=host
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=8g
    -p "${PORT}:${PORT}"
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8
    -e HF_HOME=/hf-cache -e HF_HUB_OFFLINE=1
    -e "VLLM_USE_DEEP_GEMM=${VLLM_USE_DEEP_GEMM}"
    "${docker_env[@]}"
    -v "${HF_CACHE}:/hf-cache"
    -v "${VLLM_CACHE}:/root/.cache/vllm"
    "${HEALTH_ARGS[@]}"
    "$VLLM_IMAGE"
  )

  if [ "${RUNNER}" = "pooling" ]; then
    # -- embedding / pooling model: no chat- or generate-only flags ------- #
    local dtype=() eager=()
    [ -n "${DTYPE:-}" ] && [ "${DTYPE}" != "auto" ] && dtype=(--dtype "${DTYPE}")
    [ "${ENFORCE_EAGER:-0}" = "1" ] && eager=(--enforce-eager)
    VLLM_ARGS=(
      vllm serve "${VLLM_MODEL_PATH:-$MODEL}"
      --served-model-name "$SERVED_NAME"
      --host "$HOST" --port "$PORT"
      --runner pooling
      --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      "${dtype[@]}"
      "${eager[@]}"
    )
    VISION_DESC="n/a (pooling runner)"
  else
    local spec=() vision=() prefix=() eager=()
    if [ "${MTP_TOKENS}" -gt 0 ]; then
      spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
    fi
    case "${PREFIX_CACHING:-}" in
      1) prefix=(--enable-prefix-caching) ;;
      0) prefix=(--no-enable-prefix-caching) ;;
    esac
    # ENFORCE_EAGER was only wired in the pooling branch before 2026-09-10 --
    # a silent no-op for the generate models despite the header doc's claim.
    [ "${ENFORCE_EAGER:-0}" = "1" ] && eager=(--enforce-eager)
    if [ "${LANGUAGE_MODEL_ONLY}" = "1" ]; then
      vision=(--language-model-only)
      VISION_DESC="off (--language-model-only)"
    else
      vision=(--limit-mm-per-prompt "{\"image\":${MM_IMAGE_LIMIT},\"video\":${MM_VIDEO_LIMIT}}")
      [ -n "${MM_PROCESSOR_KWARGS:-}" ] && vision+=(--mm-processor-kwargs "${MM_PROCESSOR_KWARGS}")
      VISION_DESC="on  (image<=${MM_IMAGE_LIMIT}, video<=${MM_VIDEO_LIMIT})"
    fi

    local reason=() tools=() trc=()
    [ -n "${REASONING_PARSER:-}" ] && reason=(--reasoning-parser "${REASONING_PARSER}")
    [ -n "${TOOL_CALL_PARSER:-}" ] && tools=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
    [ "${TRUST_REMOTE_CODE:-1}" = "1" ] && trc=(--trust-remote-code)

    VLLM_ARGS=(
      vllm serve "${VLLM_MODEL_PATH:-$MODEL}"
      --served-model-name "$SERVED_NAME"
      --host "$HOST" --port "$PORT"
      "${trc[@]}"
      --max-model-len "$MAX_MODEL_LEN"
      --max-num-seqs "$MAX_NUM_SEQS"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
      --kv-cache-dtype "$KV_CACHE_DTYPE"
      "${spec[@]}"
      "${vision[@]}"
      "${prefix[@]}"
      "${eager[@]}"
      "${reason[@]}"
      "${tools[@]}"
      --tensor-parallel-size 1
    )
  fi

  if [ -n "${EXTRA_VLLM_ARGS:-}" ]; then
    # shellcheck disable=SC2206  # intentional word-splitting of a flat flag list
    VLLM_ARGS+=( ${EXTRA_VLLM_ARGS} )
  fi
}

banner() {
  local gb; gb=$(python3 -c "print(round(${GPU_MEM_UTIL}*${BOX_UNIFIED_MEM_GB}))" 2>/dev/null || echo "?")
  if [ "${RUNNER:-generate}" = "pooling" ]; then
    cat <<EOF
==========================================================
 ${SLUG}   [${1}]
==========================================================
  model        : ${MODEL}$( [ -n "${VLLM_MODEL_PATH:-}" ] && echo "  (served from local path: ${VLLM_MODEL_PATH})" )
  image        : ${VLLM_IMAGE}
  runner       : pooling   (embeddings — POST /v1/embeddings)
  context      : ${MAX_MODEL_LEN} tokens
  dtype        : ${DTYPE:-auto}
  gpu mem util : ${GPU_MEM_UTIL}   (~${gb} GB of ${BOX_UNIFIED_MEM_GB} GB unified)
  enforce eager: $( [ "${ENFORCE_EAGER:-0}" = "1" ] && echo yes || echo no )
  extra args   : ${EXTRA_VLLM_ARGS:-(none)}
  docker env   : ${DOCKER_ENV:-(none)}
  container    : ${CONTAINER}
  endpoint     : http://${HOST}:${PORT}/v1   (served as '${SERVED_NAME}')
==========================================================
EOF
    return
  fi
  cat <<EOF
==========================================================
 ${SLUG}   [${1}]
==========================================================
  model        : ${MODEL}$( [ -n "${VLLM_MODEL_PATH:-}" ] && echo "  (served from local path: ${VLLM_MODEL_PATH})" )
  image        : ${VLLM_IMAGE}
  context      : ${MAX_MODEL_LEN} tokens
  max seqs     : ${MAX_NUM_SEQS}
  gpu mem util : ${GPU_MEM_UTIL}   (~${gb} GB of ${BOX_UNIFIED_MEM_GB} GB unified)
  kv cache     : ${KV_CACHE_DTYPE}
  MTP          : ${MTP_TOKENS} draft token(s)$( [ "${MTP_TOKENS}" -eq 0 ] && echo '  (disabled)' )
  vision       : ${VISION_DESC}
  prefix cache : $( case "${PREFIX_CACHING:-}" in 1) echo "on";; 0) echo "off (forced)";; *) echo "vLLM default";; esac )
  enforce eager: $( [ "${ENFORCE_EAGER:-0}" = "1" ] && echo "yes  (torch.compile OFF)" || echo "no" )
  extra args   : ${EXTRA_VLLM_ARGS:-(none)}
  docker env   : ${DOCKER_ENV:-(none)}
  container    : ${CONTAINER}
  endpoint     : http://${HOST}:${PORT}/v1   (served as '${SERVED_NAME}')
==========================================================
EOF
}

endpoint_block() {
  local ip; ip="$(lan_ip)"
  if [ "${RUNNER:-generate}" = "pooling" ]; then
    cat <<EOF
==========================================================
 OpenAI-compatible API${1:+  ($1)}
==========================================================
  base URL (local)  : http://localhost:${PORT}/v1
$( [ -n "$ip" ] && echo "  base URL (network): http://${ip}:${PORT}/v1" )
  embeddings        : POST  <base URL>/embeddings
  model id          : ${SERVED_NAME}
  auth              : none (open) -- firewall + IP whitelist is the access control
  health            : GET   http://localhost:${PORT}/health   (no auth)
==========================================================
EOF
    return
  fi
  cat <<EOF
==========================================================
 OpenAI-compatible API${1:+  ($1)}
==========================================================
  base URL (local)  : http://localhost:${PORT}/v1
$( [ -n "$ip" ] && echo "  base URL (network): http://${ip}:${PORT}/v1" )
  chat completions  : POST  <base URL>/chat/completions
  model id          : ${SERVED_NAME}
  auth              : none (open) -- firewall + IP whitelist is the access control
  health            : GET   http://localhost:${PORT}/health   (no auth)
  vision            : ${VISION_DESC}
==========================================================
EOF
}

wait_health() {
  local deadline; deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  echo -n "loading (weights + torch.compile + CUDA-graph; +~2 min for the vision encoder) "
  while :; do
    [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" || true)" = "200" ] \
      && { echo " ready"; return 0; }
    if ! d ps -q --filter "name=^${CONTAINER}$" --filter status=running | grep -q .; then
      echo " FAILED"; echo "--- last 40 log lines ---"; d logs --tail 40 "$CONTAINER"; return 1
    fi
    [ "$(date +%s)" -ge "$deadline" ] && { echo " TIMEOUT (${STARTUP_TIMEOUT}s)"; return 1; }
    echo -n "."; sleep 5
  done
}

# Boot race: vLLM sizes its KV cache from the memory that is free while it
# profiles. On unified memory a sibling still loading its weights at that
# moment is counted against this model ("Available KV cache memory: -22 GiB",
# seen on two boots), so the start fails. Start one model at a time instead:
# wait while another vLLM container is still loading (health "starting"), or
# an llm-vllm@ instance whose name sorts before ours is (re)starting but has
# no container yet. Name order keeps it cycle-free (a container only exists
# once its unit stopped waiting); capped so a stuck sibling cannot block us.
# A sibling that just turned healthy still allocates for a few seconds, which
# again shrinks our KV cache (8 instead of 14 GiB for gemma), hence the settle.
WAIT_TURN_MAX=600
WAIT_TURN_SETTLE=30
wait_turn() {
  local deadline busy waited=
  deadline=$(( $(date +%s) + WAIT_TURN_MAX ))
  while :; do
    busy="$(d ps --filter name=^vllm- --filter health=starting --format '{{.Names}}' 2>/dev/null \
      | grep -vx "$CONTAINER" | tr '\n' ' ' || true)"
    busy+="$(systemctl list-units --plain --no-legend --state=activating,deactivating 'llm-vllm@*' 2>/dev/null \
      | awk -v me="llm-vllm@${SLUG}.service" '$1 < me {printf "%s ", $1}' || true)"
    if [ -z "$busy" ]; then
      [ -n "$waited" ] && sleep "$WAIT_TURN_SETTLE"
      return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "still loading after ${WAIT_TURN_MAX}s: ${busy}-- starting anyway" >&2
      return 0
    fi
    echo "waiting for ${busy}to finish loading ..."
    waited=1
    sleep 10
  done
}

preflight() {
  if ! d image inspect "$VLLM_IMAGE" >/dev/null 2>&1; then
    echo "pulling $VLLM_IMAGE (~15 GB, one-time) ..."
    d pull "$VLLM_IMAGE"
  fi
  local wd="${HF_CACHE}/hub/models--${MODEL//\//--}"
  # a local VLLM_MODEL_PATH under /hf-cache (e.g. a converted copy, see
  # mistral-small4-heretic) is what actually gets served -- check that instead
  case "${VLLM_MODEL_PATH:-}" in
    /hf-cache/*) wd="${HF_CACHE}${VLLM_MODEL_PATH#/hf-cache}" ;;
  esac
  if [ ! -d "$wd" ]; then
    echo "error: weights for '${MODEL}' not found at ${wd}" >&2
    echo "       HF_HOME=${HF_CACHE} HF_TOKEN=... ${LLM_ROOT}/.hf-venv/bin/hf download ${MODEL}" >&2
    exit 1
  fi
  mkdir -p "$VLLM_CACHE"
}

# -- systemd (template unit: one file, one instance per model slug) ------ #
UNIT_TEMPLATE_PATH="/etc/systemd/system/llm-vllm@.service"

install_service() {
  # the account the service runs as: whoever invoked us, even under `sudo`.
  local svc_user="${SUDO_USER:-$(id -un)}"
  if [ "$svc_user" = "root" ]; then
    echo "error: run this WITHOUT 'sudo' — the script sudo's only the bits that" >&2
    echo "       need it, and needs to know which non-root user runs the service." >&2
    echo "       just:   ${INVOKED_AS} install-service" >&2
    exit 1
  fi
  if ! id -nG "$svc_user" | tr ' ' '\n' | grep -qx docker; then
    echo "error: user '$svc_user' is not in the 'docker' group." >&2
    echo "       sudo usermod -aG docker $svc_user   # then a fresh login shell" >&2
    exit 1
  fi
  echo "writing ${UNIT_TEMPLATE_PATH}  (User=${svc_user}, sudo) ..."
  sudo tee "$UNIT_TEMPLATE_PATH" >/dev/null <<UNIT
[Unit]
Description=vLLM server: %i
Requires=docker.service
After=${BOX_AFTER_UNITS}
Wants=network-online.target
StartLimitIntervalSec=1800
StartLimitBurst=5

[Service]
Type=exec
User=${svc_user}
Group=docker
# systemd is the single supervisor: 'run' execs 'docker run --rm' in the
# foreground, so there is no container --restart policy to race with on boot.
ExecStartPre=-/usr/bin/docker rm -f vllm-%i
ExecStart=${COMMON_DIR}/serve.sh ${LLM_ROOT}/models/%i run
ExecStop=/usr/bin/docker stop -t 30 vllm-%i
# hold the unit 'activating' until the container reports healthy (model
# serving); give up as soon as the main process is gone, so a failed start
# restarts after RestartSec instead of sitting out the whole timeout
ExecStartPost=/usr/bin/timeout ${STARTUP_TIMEOUT} /bin/sh -c 'until [ "\$(/usr/bin/docker inspect -f "{{.State.Health.Status}}" vllm-%i 2>/dev/null)" = healthy ]; do kill -0 "\$MAINPID" 2>/dev/null || exit 1; sleep 5; done'
Restart=always
RestartSec=15
TimeoutStartSec=${STARTUP_TIMEOUT}
TimeoutStopSec=45

[Install]
WantedBy=multi-user.target
UNIT
  sudo systemctl daemon-reload
  sudo systemctl enable --now "llm-vllm@${SLUG}"
  echo
  echo "enabled + started 'llm-vllm@${SLUG}'.  comes up on every boot now."
  echo "  systemctl status llm-vllm@${SLUG}"
  echo "  journalctl -u llm-vllm@${SLUG} -f"
}

uninstall_service() {
  echo "disabling + stopping 'llm-vllm@${SLUG}'  (sudo) ..."
  sudo systemctl disable --now "llm-vllm@${SLUG}" 2>/dev/null || true
  sudo systemctl reset-failed "llm-vllm@${SLUG}" 2>/dev/null || true
  d rm -f "vllm-${SLUG}" >/dev/null 2>&1 || true
  # drop the shared template only if no other instance still uses it
  if [ -f "$UNIT_TEMPLATE_PATH" ] && \
     ! systemctl list-units --all --plain --no-legend 'llm-vllm@*' 2>/dev/null | grep -q .; then
    sudo rm -f "$UNIT_TEMPLATE_PATH"
    sudo systemctl daemon-reload
    echo "removed the llm-vllm@ template too (no instances left)."
  else
    echo "(llm-vllm@.service template kept — other model instances still use it)"
  fi
  echo "start it ad-hoc again with:  ${INVOKED_AS} start"
}

# true if this model's systemd instance is enabled or running — in that case
# systemd owns the container and bare start/stop must not touch it.
systemd_owns() {
  [ -f "$UNIT_TEMPLATE_PATH" ] || return 1
  local st; st="$(systemctl is-active "llm-vllm@${SLUG}" 2>/dev/null)"
  [ "$st" = active ] || [ "$st" = activating ] || \
    [ "$(systemctl is-enabled "llm-vllm@${SLUG}" 2>/dev/null)" = enabled ]
}

# -- subcommand dispatcher ---------------------------------------------- #
main() {
  case "${1:-start}" in
    stop)
      if systemd_owns; then
        echo "llm-vllm@${SLUG} is managed by systemd — use:  sudo systemctl stop llm-vllm@${SLUG}" >&2
        exit 1
      fi
      if d ps -aq --filter "name=^${CONTAINER}$" | grep -q .; then
        d rm -f "$CONTAINER" >/dev/null && echo "stopped $CONTAINER"
      else echo "not running"; fi
      ;;
    logs)   d logs -f "$CONTAINER" ;;
    install-service)   install_service ;;
    uninstall-service) uninstall_service ;;
    status)
      if [ -f "$UNIT_TEMPLATE_PATH" ]; then
        echo "systemd : llm-vllm@${SLUG}  ->  $(systemctl is-enabled "llm-vllm@${SLUG}" 2>/dev/null || echo -) / $(systemctl is-active "llm-vllm@${SLUG}" 2>/dev/null || echo -)"
      else
        echo "systemd : template not installed  (serve.sh <model> install-service)"
      fi
      d ps --filter "name=^${CONTAINER}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
      local ip; ip="$(lan_ip)"
      if [ "$(curl -s -m 3 -o /dev/null -w '%{http_code}' "http://localhost:${PORT}/health" || true)" = "200" ]; then
        echo "health : OK"
        echo "API    : http://localhost:${PORT}/v1${ip:+   |   http://${ip}:${PORT}/v1}"
        curl -fsS -m 3 "http://localhost:${PORT}/v1/models" 2>/dev/null \
          | python3 -m json.tool 2>/dev/null || true
      else
        echo "health : not responding on :${PORT} (still loading, or not started)"
      fi
      ;;
    start|run)
      local mode="${1:-start}"
      if [ "$mode" = start ] && systemd_owns; then
        echo "llm-vllm@${SLUG} is managed by systemd — use:  sudo systemctl start llm-vllm@${SLUG}" >&2
        echo "(or  ${INVOKED_AS} uninstall-service  to hand control back to this script)" >&2
        exit 1
      fi
      assemble_args
      preflight
      banner "$mode"
      d rm -f "$CONTAINER" >/dev/null 2>&1 || true
      if [ "$mode" = "run" ]; then
        # foreground: the systemd unit (or you) supervises this directly.
        # no -d, no --restart (the unit's Restart=always owns that), --rm.
        endpoint_block "reachable once the model finishes loading"
        wait_turn
        echo "exec: docker run --rm ${CONTAINER}"
        exec_docker run --rm "${RUN_COMMON[@]}" "${VLLM_ARGS[@]}"
      fi
      d run -d --restart unless-stopped "${RUN_COMMON[@]}" "${VLLM_ARGS[@]}"
      wait_health || exit 1
      echo
      endpoint_block
      if [ "${RUNNER}" = "pooling" ]; then
        cat <<EOF

  test:        curl -s http://localhost:${PORT}/v1/embeddings \\
                 -H 'Content-Type: application/json' \\
                 -d '{"model":"${SERVED_NAME}","input":["Hallo Welt","the quick brown fox"]}'
  pooling/norm: read from the model config (1_Pooling/, sentence_bert_config.json)

  persistent across reboots:   ${INVOKED_AS} install-service
  logs / stop / status:        ${INVOKED_AS} <that subcommand>
EOF
      else
        cat <<EOF

  text test:   curl -s http://localhost:${PORT}/v1/chat/completions \\
                 -H 'Content-Type: application/json' \\
                 -d '{"model":"${SERVED_NAME}","messages":[{"role":"user","content":"hi"}]}'
  vision/OCR:  content = [ {"type":"text",...}, {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}} ]
  no thinking: add   "chat_template_kwargs": {"enable_thinking": false}

  persistent across reboots:   ${INVOKED_AS} install-service
  logs / stop / status:        ${INVOKED_AS} <that subcommand>
EOF
      fi
      ;;
    *)
      echo "usage: serve <model-dir> [start|run|stop|logs|status|install-service|uninstall-service]" >&2
      exit 2 ;;
  esac
}
