"""Shared: read common/box.env + models/<slug>/model.env for the bench tools."""
from __future__ import annotations
import os
import re
from pathlib import Path

_ENV_RE = re.compile(r'^\s*(?:: *"\$\{(\w+):=(.*?)\}"|(\w+)=(.*?))\s*(?:#.*)?$')


def parse_env_file(path: Path) -> dict:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _ENV_RE.match(line)
        if not m:
            continue
        k = m.group(1) or m.group(3)
        v = (m.group(2) if m.group(1) else m.group(4)).strip().strip('"').strip("'")
        out[k] = v
    return out


def load_model_env(model_dir: Path) -> dict:
    """box.env then model.env then the real environment (matches := semantics)."""
    common = model_dir.parent.parent / "common"
    env = {}
    env.update(parse_env_file(common / "box.env"))
    env.update(parse_env_file(model_dir / "model.env"))
    for k in list(env):
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def resolve_snapshot(hf_cache: Path, repo_id: str) -> str | None:
    """Newest local snapshot dir for a HF repo id, as a /hf-cache/... container path."""
    base = Path(hf_cache) / f"hub/models--{repo_id.replace('/', '--')}/snapshots"
    if not base.is_dir():
        return None
    snaps = [d for d in base.iterdir() if d.is_dir()]
    if not snaps:
        return None
    newest = max(snaps, key=lambda d: d.stat().st_mtime)
    return str(newest).replace(str(hf_cache), "/hf-cache")
