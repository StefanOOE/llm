#!/bin/bash
# =============================================================================
#  setup-firewall.sh — ufw + IP whitelist for this box's exposed ports.
#  Idempotent by RECONCILE, not skip: on every run it deletes its own
#  previously-appended marker block (if present) and re-appends the current
#  full desired block below -- so this script is the single source of truth
#  for "what the firewall should look like"; editing the whitelist here and
#  re-running always converges, rather than silently no-op'ing because an
#  older version already ran once. Safe to re-run any time.
#
#  The client IPs are NOT in this file: they come from the git-ignored
#  common/firewall.local.env (copy common/firewall.local.env.example and
#  fill it in). The script aborts if that file or a value is missing.
#
#  Needs root (editing /etc/ufw/*.rules, `ufw enable`/`ufw allow`):
#
#      sudo bash common/setup-firewall.sh
#      bash common/setup-firewall.sh --dry-run   # print the rules, change nothing
#
#  WHY two different mechanisms (see README "Access control"):
#  8000/8001/8002/8004/8501 are all Docker-published container ports. Docker's
#  own DOCKER-USER/DOCKER iptables chains handle forwarded/published-port
#  traffic BEFORE ufw's normal INPUT-chain rules ever run -- a plain
#  `ufw allow from <ip> to any port <port>` silently does nothing for a
#  dockerized port. Only SSH (22, a plain host process) is reachable via a
#  normal ufw allow rule. The real enforcement for the Docker ports is the
#  DOCKER-USER chain block below, appended into ufw's after.rules/
#  after6.rules (which ufw-init re-applies on every enable/reload/boot, and
#  which Docker's own chain bootstrap respects rather than overwrites).
#
#  To whitelist a NEW device or port: add another `-s <ip>/32 --dport <port>
#  -j RETURN` line before the matching `--dport <port> -j DROP` line for
#  that port (order matters -- RETURN must come first) and re-run.
# =============================================================================
set -euo pipefail

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV="${FIREWALL_LOCAL_ENV:-${HERE}/firewall.local.env}"
if [ ! -r "${LOCAL_ENV}" ]; then
  echo "ERROR: ${LOCAL_ENV} not found." >&2
  echo "       cp ${HERE}/firewall.local.env.example ${HERE}/firewall.local.env" >&2
  echo "       and set WORKSTATION_IP / HARNESS_IP for this network." >&2
  exit 1
fi
# shellcheck source=/dev/null
. "${LOCAL_ENV}"
for v in WORKSTATION_IP HARNESS_IP; do
  if ! [[ "${!v:-}" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
    echo "ERROR: ${v} is unset or not an IPv4 address in ${LOCAL_ENV}" >&2
    exit 1
  fi
done

MARKER="${FIREWALL_MARKER:-LLM-SERVER-WHITELIST}"
# Blocks written under an older marker name are stripped too (space-separated).
LEGACY_MARKERS="${FIREWALL_LEGACY_MARKERS:-}"
TS="$(date +%Y%m%d-%H%M%S)"

# WORKSTATION_IP: SSH/SCP + DGX Dashboard (:8501) + gb10-rendering bridge (:8002)
# HARNESS_IP:     LLM APIs (:8000/:8001/:8004) + gb10-rendering bridge (:8002, the agent harness calls /generate)
# Any other host gets nothing.

rules_v4() {
cat <<EOF

# BEGIN ${MARKER}
*filter
:DOCKER-USER - [0:0]
-A DOCKER-USER -p tcp -s ${HARNESS_IP}/32 --dport 8000 -j RETURN
-A DOCKER-USER -p tcp -s ${HARNESS_IP}/32 --dport 8001 -j RETURN
-A DOCKER-USER -p tcp -s ${HARNESS_IP}/32 --dport 8002 -j RETURN
-A DOCKER-USER -p tcp -s ${HARNESS_IP}/32 --dport 8004 -j RETURN
-A DOCKER-USER -p tcp -s ${WORKSTATION_IP}/32 --dport 8002 -j RETURN
-A DOCKER-USER -p tcp -s ${WORKSTATION_IP}/32 --dport 8501 -j RETURN
-A DOCKER-USER -p tcp --dport 8000 -j DROP
-A DOCKER-USER -p tcp --dport 8001 -j DROP
-A DOCKER-USER -p tcp --dport 8002 -j DROP
# 8003: gb10-code removed 2026-10-11 -- no whitelist, kept closed
-A DOCKER-USER -p tcp --dport 8003 -j DROP
-A DOCKER-USER -p tcp --dport 8004 -j DROP
-A DOCKER-USER -p tcp --dport 8501 -j DROP
-A DOCKER-USER -j RETURN
COMMIT
# END ${MARKER}
EOF
}

rules_v6() {
cat <<EOF

# BEGIN ${MARKER}
*filter
:DOCKER-USER - [0:0]
-A DOCKER-USER -p tcp --dport 8000 -j DROP
-A DOCKER-USER -p tcp --dport 8001 -j DROP
-A DOCKER-USER -p tcp --dport 8002 -j DROP
# 8003: gb10-code removed 2026-10-11 -- no whitelist, kept closed
-A DOCKER-USER -p tcp --dport 8003 -j DROP
-A DOCKER-USER -p tcp --dport 8004 -j DROP
-A DOCKER-USER -p tcp --dport 8501 -j DROP
-A DOCKER-USER -j RETURN
COMMIT
# END ${MARKER}
EOF
}

if [ "${DRY_RUN}" = 1 ]; then
  echo "== DRY RUN: nothing is changed =="
  echo "== would strip blocks: ${MARKER} ${LEGACY_MARKERS}"
  echo "== /etc/ufw/after.rules would get:"; rules_v4
  echo "== /etc/ufw/after6.rules would get:"; rules_v6
  echo "== would run: ufw allow from ${WORKSTATION_IP} to any port 22 proto tcp; ufw default deny incoming; ufw --force enable; ufw reload"
  exit 0
fi

echo "== backing up ufw after-rules =="
cp -a /etc/ufw/after.rules  "/etc/ufw/after.rules.bak-${TS}"
cp -a /etc/ufw/after6.rules "/etc/ufw/after6.rules.bak-${TS}"

# reconcile: strip any previous version of our block, then re-append current
for m in ${MARKER} ${LEGACY_MARKERS}; do
  sed -i "/# BEGIN ${m}/,/# END ${m}/d" /etc/ufw/after.rules
  sed -i "/# BEGIN ${m}/,/# END ${m}/d" /etc/ufw/after6.rules
done

echo "== DOCKER-USER chain (IPv4): whitelist harness+workstation -> gb10-rendering, harness -> LLM APIs, workstation -> dashboard =="
rules_v4 >> /etc/ufw/after.rules

echo "== DOCKER-USER chain (IPv6): no IPv6 whitelist needed, drop these ports outright =="
rules_v6 >> /etc/ufw/after6.rules

echo "== ufw: SSH allow from the workstation only =="
ufw allow from "${WORKSTATION_IP}" to any port 22 proto tcp comment 'SSH from workstation'

echo "== ufw: explicit default deny incoming (matches existing policy) =="
ufw default deny incoming

echo "== enabling ufw =="
ufw --force enable

echo "== reloading to apply after.rules/after6.rules (DOCKER-USER) immediately =="
ufw reload

echo
echo "================= ufw status ================="
ufw status verbose
echo
echo "================= DOCKER-USER chain (IPv4) ================="
iptables -L DOCKER-USER -n -v
echo
echo "================= DOCKER-USER chain (IPv6) ================="
ip6tables -L DOCKER-USER -n -v
