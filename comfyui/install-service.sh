#!/bin/bash
# Installs the comfyui.service systemd unit (gb10-rendering). Needs plain
# sudo (writing to /etc/systemd/system/ isn't in the passwordless-sudo
# systemctl-only scope) --
#
#     sudo bash comfyui/install-service.sh
#
# Single instance, not a @-template like llm-vllm@ -- there's only ever one
# rendering backend, no interchangeable-checkpoint toggle needed.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
SVC_USER="${SUDO_USER:-$(id -un)}"

if [ "$SVC_USER" = "root" ]; then
  echo "error: run this WITHOUT plain 'sudo -i' as root first -- just: sudo bash comfyui/install-service.sh" >&2
  exit 1
fi

cat > /etc/systemd/system/comfyui.service <<UNIT
[Unit]
Description=ComfyUI: gb10-rendering (Qwen-Image-2512)
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target
StartLimitIntervalSec=1800
StartLimitBurst=5

[Service]
Type=exec
User=${SVC_USER}
Group=docker
ExecStartPre=-/usr/bin/docker rm -f comfyui-gb10-rendering
ExecStart=${HERE}/serve.sh run
ExecStop=/usr/bin/docker stop -t 30 comfyui-gb10-rendering
ExecStartPost=/usr/bin/timeout 600 /bin/sh -c 'until [ "\$(/usr/bin/docker inspect -f "{{.State.Health.Status}}" comfyui-gb10-rendering 2>/dev/null)" = healthy ]; do sleep 5; done'
Restart=always
RestartSec=15
TimeoutStartSec=600
TimeoutStopSec=45

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
echo "installed comfyui.service (User=${SVC_USER}). Not started/enabled yet -- that's left to you:"
echo "  sudo systemctl start comfyui     # start now, ad-hoc"
echo "  sudo systemctl enable comfyui    # autostart on boot (NOT done by default -- see memory-budget note in comfyui/notes.md before enabling)"
