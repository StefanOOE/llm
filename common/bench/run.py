#!/usr/bin/env python3
"""
Container-based vLLM benchmark runner for this box.

    sg docker -c ".hf-venv/bin/python common/bench/run.py --model models/<slug> --suite <name>"

Reads  common/box.env + models/<slug>/model.env  (image, caches, repo id, served
name, port) and  models/<slug>/bench/matrix.yaml  (the sweep: configs, workloads,
suites). Launches one container per config, drives it with `vllm bench serve` per
paired workload, writes  models/<slug>/bench/results/<suite>_<ts>/ .

Flags:
    --model DIR        the model directory (required)
    --suite NAME       suite from that model's matrix.yaml (default: overnight)
    --only  a,b,c      restrict to these config names
    --dry-run          print the plan and exit
    --keep-going       don't abort the suite if one config fails to start
    --resume DIR       skip (config,workload) pairs already 'ok' in DIR/summary.csv

The model's serving port must be FREE (stop the running server first:
  systemctl stop llm-vllm@<slug>   or   models/<slug>/serve stop).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

from _env import load_model_env, resolve_snapshot as _resolve_snapshot

# populated by load_env() from box.env + model.env
HF_CACHE_HOST = Path("/home/ss/models/hf-cache")
VLLM_CACHE_HOST = Path("/home/ss/models/vllm-cache")
VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.07-py3"
SERVED = "model"
PORT = 8000
CONTAINER = "bench-model"
BASE_URL = "http://localhost:8000"
MODEL_DIR = Path(".")


def load_env(model_dir: Path) -> dict:
    env = load_model_env(model_dir)
    global HF_CACHE_HOST, VLLM_CACHE_HOST, VLLM_IMAGE, SERVED, PORT, CONTAINER, BASE_URL, MODEL_DIR
    MODEL_DIR = model_dir
    HF_CACHE_HOST = Path(env.get("HF_CACHE", HF_CACHE_HOST))
    VLLM_CACHE_HOST = Path(env.get("VLLM_CACHE", VLLM_CACHE_HOST))
    VLLM_IMAGE = env.get("VLLM_IMAGE", VLLM_IMAGE)
    SERVED = env.get("SERVED_NAME", model_dir.name)
    PORT = int(env.get("PORT", 8000))
    CONTAINER = f"bench-{model_dir.name}"
    BASE_URL = f"http://localhost:{PORT}"
    return env


def resolve_snapshot(repo_id: str) -> str | None:
    return _resolve_snapshot(HF_CACHE_HOST, repo_id)


# --------------------------------------------------------------------------- #
# matrix loading / resolution
# --------------------------------------------------------------------------- #
def load_matrix(model_dir: Path) -> dict:
    return yaml.safe_load((model_dir / "bench" / "matrix.yaml").read_text())


def resolve_suite(mx: dict, suite: str) -> list[tuple[dict, list[dict]]]:
    d = mx["defaults"]
    suites = mx["suites"]
    if suite not in suites:
        sys.exit(f"unknown suite {suite!r}; have {list(suites)}")
    spec = suites[suite]
    pairs_raw: list = []
    if isinstance(spec, dict) and "include" in spec:
        for sub in spec["include"]:
            pairs_raw += suites[sub]
    else:
        pairs_raw = spec

    out = []
    for cfg_name, wl_names in pairs_raw:
        cfg = dict(d["config_defaults"])
        cfg.update(mx["configs"][cfg_name])
        cfg["name"] = cfg_name
        wls = []
        for wn in wl_names:
            wl = dict(d["workload_defaults"])
            wl.update(mx["workloads"][wn])
            wl["name"] = wn
            wls.append(wl)
        out.append((cfg, wls))
    return out


# --------------------------------------------------------------------------- #
# docker helpers
# --------------------------------------------------------------------------- #
def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def docker_rm():
    sh(["docker", "rm", "-f", CONTAINER])


def free_used_mb() -> int | None:
    p = sh(["free", "-m"])
    for line in p.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].rstrip(":").lower() in ("mem", "speicher"):
            return int(parts[2])
    return None


class MemSampler:
    """Sample `free -m` used-MB in a thread; expose peak and mean."""

    def __init__(self, interval=2.0):
        self.interval = interval
        self._stop = threading.Event()
        self._samples: list[int] = []
        self._t: threading.Thread | None = None

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                v = free_used_mb()
                if v is not None:
                    self._samples.append(v)
                self._stop.wait(self.interval)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t:
            self._t.join(timeout=5)

    @property
    def peak(self):
        return max(self._samples) if self._samples else None

    @property
    def mean(self):
        return round(sum(self._samples) / len(self._samples)) if self._samples else None


def build_serve_cmd(mx: dict, cfg: dict) -> tuple[list[str], dict]:
    """Return (docker-run argv, resolved model info dict with 'snapshot')."""
    d = mx["defaults"]
    model = dict(d["models"][cfg["model"]])
    snap = model.get("snapshot")
    if not snap:
        sys.exit(f"model {cfg['model']!r} ({model.get('path')}) not downloaded")

    env = dict(d.get("env", {}))
    env["HF_HOME"] = "/hf-cache"

    serve = [
        "vllm", "serve", snap,
        "--served-model-name", SERVED,
        "--host", "0.0.0.0", "--port", str(PORT),
        "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"]),
        "--max-model-len", str(cfg["max_model_len"]),
        "--max-num-seqs", str(cfg["max_num_seqs"]),
        "--kv-cache-dtype", cfg["kv_cache_dtype"],
        *d.get("common_flags", []),
    ]
    if cfg.get("spec"):
        serve += ["--speculative-config", json.dumps(cfg["spec"], separators=(",", ":"))]
    if cfg.get("prefix_caching"):
        serve += ["--enable-prefix-caching"]
    else:
        serve += ["--no-enable-prefix-caching"]
    if cfg.get("max_num_batched_tokens"):
        serve += ["--max-num-batched-tokens", str(cfg["max_num_batched_tokens"])]
    serve += list(cfg.get("extra_flags", []))

    argv = [
        "docker", "run", "-d", "--name", CONTAINER,
        "--gpus", "all", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "--shm-size=8g",
        "-p", f"{PORT}:{PORT}",
        "-v", f"{HF_CACHE_HOST}:/hf-cache",
        "-v", f"{VLLM_CACHE_HOST}:/root/.cache/vllm",
    ]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    argv += [VLLM_IMAGE, *serve]
    return argv, model


SERVER_LOG_PATTERNS = {
    "model_load_gib": r"Model loading took ([\d.]+) GiB",
    "model_load_s": r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds",
    "kv_cache_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_cache_tokens": r"GPU KV cache size: ([\d,]+) tokens",
    "max_concurrency_x": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
}


def parse_server_log(text: str) -> dict:
    out: dict = {}
    for key, pat in SERVER_LOG_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1).replace(",", "")
            out[key] = float(val) if "." in val else int(val)
    return out


def wait_health(timeout_s: int) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if sh(["docker", "ps", "-q", "--filter", f"name={CONTAINER}",
               "--filter", "status=running"]).stdout.strip() == "":
            return False  # container exited
        c = sh(["curl", "-s", "-m", "3", "-o", "/dev/null", "-w", "%{http_code}",
                f"{BASE_URL}/health"])
        if c.stdout.strip() == "200":
            return True
        time.sleep(5)
    return False


# --------------------------------------------------------------------------- #
# workload execution
# --------------------------------------------------------------------------- #
def run_workload(mx: dict, cfg: dict, wl: dict, model: dict, outdir: Path) -> dict:
    d = mx["defaults"]
    rf = f"{cfg['name']}__{wl['name']}.json"
    bench = [
        "docker", "exec", "-e", "LANG=C.UTF-8", CONTAINER,
        "vllm", "bench", "serve",
        "--backend", "openai-chat",
        "--base-url", BASE_URL, "--endpoint", "/v1/chat/completions",
        "--model", SERVED,
        "--tokenizer", model["snapshot"],
        "--dataset-name", "random",
        "--random-input-len", str(wl["input_len"]),
        "--random-output-len", str(wl["output_len"]),
    ]
    # optional: fixed shared prefix (+ suffix-length jitter) to emulate a
    # repeated large system prompt/tool schema -- both are no-ops (0/absent)
    # for every workload that doesn't set them.
    if wl.get("prefix_len"):
        bench += ["--random-prefix-len", str(wl["prefix_len"])]
    if wl.get("random_range_ratio"):
        bench += ["--random-range-ratio", str(wl["random_range_ratio"])]
    bench += [
        "--num-prompts", str(wl["num_prompts"]),
        "--max-concurrency", str(wl["concurrency"]),
        "--request-rate", str(wl["request_rate"]),
        "--num-warmups", str(wl["warmups"]),
        "--seed", "0",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,90,99",
        "--save-result", "--result-dir", "/hf-cache/bench-tmp",
        "--result-filename", rf,
    ]
    if wl.get("ignore_eos", True):
        bench.append("--ignore-eos")

    sh(["docker", "exec", CONTAINER, "rm", "-f", f"/hf-cache/bench-tmp/{rf}"])
    print(f"    -> {wl['name']}: in={wl['input_len']} out={wl['output_len']} "
          f"n={wl['num_prompts']} conc={wl['concurrency']}", flush=True)

    t0 = time.time()
    with MemSampler() as ms:
        proc = subprocess.run(bench, text=True, capture_output=True, timeout=7200)
    wall = time.time() - t0

    raw = sh(["docker", "exec", CONTAINER, "cat", f"/hf-cache/bench-tmp/{rf}"]).stdout
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        (outdir / f"{cfg['name']}__{wl['name']}.FAILED.log").write_text(
            proc.stdout + "\n---STDERR---\n" + proc.stderr)
        return {"config": cfg["name"], "workload": wl["name"], "status": "bench_failed",
                "wall_s": round(wall, 1)}

    (outdir / rf).write_text(json.dumps(result, indent=2))
    per_pos = result.get("spec_decode_per_position_acceptance_rates") or []
    completed = result.get("completed") or 0
    failed = result.get("failed") or 0
    ok = completed > 0 and result.get("output_throughput", 0) > 0 and failed == 0
    row = {
        "config": cfg["name"],
        "workload": wl["name"],
        "status": "ok" if ok else "bench_errored",
        "model": cfg["model"],
        "max_model_len": cfg["max_model_len"],
        "max_num_seqs": cfg["max_num_seqs"],
        "kv_cache_dtype": cfg["kv_cache_dtype"],
        "gpu_mem_util": cfg["gpu_memory_utilization"],
        "spec": f"mtp{cfg['spec']['num_speculative_tokens']}" if cfg.get("spec") else "off",
        "num_spec_tokens": cfg["spec"]["num_speculative_tokens"] if cfg.get("spec") else 0,
        "prefix_caching": cfg.get("prefix_caching", False),
        "input_len": wl["input_len"],
        "output_len": wl["output_len"],
        "num_prompts": wl["num_prompts"],
        "concurrency": wl["concurrency"],
        "completed": result.get("completed"),
        "failed": result.get("failed"),
        "duration_s": round(result.get("duration", 0), 1),
        "output_tok_s": round(result.get("output_throughput", 0), 2),
        "total_tok_s": round(result.get("total_token_throughput", 0), 2),
        "req_s": round(result.get("request_throughput", 0), 4),
        "max_output_tok_s": round(result.get("max_output_tokens_per_s", 0), 2),
        "mean_ttft_ms": round(result.get("mean_ttft_ms", 0), 1),
        "p50_ttft_ms": round(result.get("median_ttft_ms", 0), 1),
        "p99_ttft_ms": round(result.get("p99_ttft_ms", 0), 1),
        "mean_tpot_ms": round(result.get("mean_tpot_ms", 0), 2),
        "p99_tpot_ms": round(result.get("p99_tpot_ms", 0), 2),
        "mean_itl_ms": round(result.get("mean_itl_ms", 0), 2),
        "mean_e2el_ms": round(result.get("mean_e2el_ms", 0), 1),
        "spec_accept_rate": round(result.get("spec_decode_acceptance_rate", 0) or 0, 4),
        "spec_accept_len": round(result.get("spec_decode_acceptance_length", 0) or 0, 3),
        "spec_pos0": round(per_pos[0], 4) if len(per_pos) > 0 else None,
        "spec_pos1": round(per_pos[1], 4) if len(per_pos) > 1 else None,
        "spec_pos2": round(per_pos[2], 4) if len(per_pos) > 2 else None,
        "spec_pos3": round(per_pos[3], 4) if len(per_pos) > 3 else None,
        "mem_peak_mb": ms.peak,
        "mem_mean_mb": ms.mean,
        "wall_s": round(wall, 1),
    }
    print(f"       {row['output_tok_s']} tok/s out | "
          f"ttft p50 {row['p50_ttft_ms']}ms | accept_len {row['spec_accept_len']}", flush=True)
    return row


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
SUMMARY_FIELDS = [
    "config", "workload", "status", "model", "max_model_len", "max_num_seqs",
    "kv_cache_dtype", "gpu_mem_util", "spec", "num_spec_tokens", "prefix_caching",
    "input_len", "output_len", "num_prompts", "concurrency",
    "completed", "failed", "duration_s",
    "output_tok_s", "total_tok_s", "req_s", "max_output_tok_s",
    "mean_ttft_ms", "p50_ttft_ms", "p99_ttft_ms",
    "mean_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "mean_e2el_ms",
    "spec_accept_rate", "spec_accept_len", "spec_pos0", "spec_pos1", "spec_pos2", "spec_pos3",
    "server_model_load_gib", "server_model_load_s", "server_kv_cache_gib",
    "server_kv_cache_tokens", "server_max_concurrency_x",
    "mem_base_mb", "mem_ready_mb", "mem_peak_mb", "mem_mean_mb",
    "wall_s",
]


def append_summary(outdir: Path, rows: list[dict]):
    csv_path = outdir / "summary.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    allrows = list(csv.DictReader((outdir / "summary.csv").open()))
    (outdir / "summary.json").write_text(json.dumps(allrows, indent=2))


def already_done(outdir: Path) -> set[tuple[str, str]]:
    p = outdir / "summary.csv"
    if not p.exists():
        return set()
    return {(r["config"], r["workload"]) for r in csv.DictReader(p.open())
            if r.get("status") == "ok"}


def resolve_all_snapshots(mx: dict):
    for name, m in mx["defaults"].get("models", {}).items():
        m["snapshot"] = resolve_snapshot(m["path"])


def port_free() -> bool:
    r = sh(["curl", "-s", "-m", "2", "-o", "/dev/null", "-w", "%{http_code}",
            f"{BASE_URL}/health"])
    return r.stdout.strip() not in ("200", "401", "403")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model directory (models/<slug>)")
    ap.add_argument("--suite", default="overnight")
    ap.add_argument("--only", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    model_dir = Path(args.model).resolve()
    if not (model_dir / "model.env").is_file():
        sys.exit(f"no model.env in {model_dir}")
    load_env(model_dir)

    mx = load_matrix(model_dir)
    resolve_all_snapshots(mx)
    plan = resolve_suite(mx, args.suite)
    if args.only:
        keep = set(args.only.split(","))
        plan = [(c, w) for c, w in plan if c["name"] in keep]

    if args.resume:
        outdir = Path(args.resume)
        done = already_done(outdir)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = model_dir / "bench" / "results" / f"{args.suite}_{stamp}"
        done = set()

    if not args.dry_run and not port_free():
        sys.exit(f"port {PORT} is in use — stop the running server first "
                 f"(systemctl stop llm-vllm@{model_dir.name}  /  {model_dir}/serve stop)")

    print(f"suite={args.suite}  outdir={outdir}")
    total = sum(len(w) for _, w in plan)
    print(f"{len(plan)} configs, {total} (config,workload) runs\n")
    for cfg, wls in plan:
        tag = "spec=" + (str(cfg["spec"]["num_speculative_tokens"]) if cfg.get("spec") else "off")
        print(f"  {cfg['name']:16s} model={cfg['model']:4s} len={cfg['max_model_len']:>6d} "
              f"seqs={cfg['max_num_seqs']:>3d} kv={cfg['kv_cache_dtype']:4s} {tag}  "
              f"<- {[w['name'] for w in wls]}")
    if args.dry_run:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    VLLM_CACHE_HOST.mkdir(parents=True, exist_ok=True)
    (HF_CACHE_HOST / "bench-tmp").mkdir(parents=True, exist_ok=True)

    n = 0
    for cfg, wls in plan:
        if not mx["defaults"]["models"][cfg["model"]]["snapshot"]:
            print(f"\n=== {cfg['name']}: model {cfg['model']!r} not downloaded, skipping ===")
            append_summary(outdir, [{"config": cfg["name"], "workload": "-",
                                     "status": "model_missing", "model": cfg["model"]}])
            continue
        pending = [w for w in wls if (cfg["name"], w["name"]) not in done]
        if not pending:
            print(f"\n=== {cfg['name']}: all workloads already done, skipping ===")
            continue

        print(f"\n=== {cfg['name']} ===", flush=True)
        docker_rm()
        mem_base = free_used_mb()
        argv, model = build_serve_cmd(mx, cfg)
        (outdir / f"{cfg['name']}.cmd").write_text(" ".join(shlex.quote(a) for a in argv))
        cp = sh(argv)
        if cp.returncode != 0:
            print(f"  docker run failed: {cp.stderr.strip()}")
            if not args.keep_going:
                sys.exit(1)
            continue

        ok = wait_health(mx["defaults"]["server_startup_timeout_s"])
        lg = sh(["docker", "logs", CONTAINER])
        logs = lg.stdout + lg.stderr
        (outdir / f"server_{cfg['name']}.log").write_text(logs)
        srv = parse_server_log(logs)
        mem_ready = free_used_mb()

        if not ok:
            print(f"  server did not become healthy; see server_{cfg['name']}.log")
            append_summary(outdir, [{"config": cfg["name"], "workload": "-",
                                     "status": "server_unhealthy",
                                     "mem_base_mb": mem_base, "mem_ready_mb": mem_ready,
                                     **{f"server_{k}": v for k, v in srv.items()}}])
            docker_rm()
            if not args.keep_going:
                sys.exit(1)
            continue

        print(f"  ready: weights {srv.get('model_load_gib','?')} GiB in "
              f"{srv.get('model_load_s','?')}s | KV {srv.get('kv_cache_gib','?')} GiB "
              f"({srv.get('kv_cache_tokens','?')} tok, {srv.get('max_concurrency_x','?')}x) | "
              f"free used {mem_base}->{mem_ready} MB", flush=True)

        rows = []
        for wl in pending:
            n += 1
            print(f"  [{n}/{total}] {cfg['name']} / {wl['name']}", flush=True)
            try:
                row = run_workload(mx, cfg, wl, model, outdir)
            except subprocess.TimeoutExpired:
                row = {"config": cfg["name"], "workload": wl["name"], "status": "timeout"}
            row.update({
                "mem_base_mb": mem_base, "mem_ready_mb": mem_ready,
                "server_model_load_gib": srv.get("model_load_gib"),
                "server_model_load_s": srv.get("model_load_s"),
                "server_kv_cache_gib": srv.get("kv_cache_gib"),
                "server_kv_cache_tokens": srv.get("kv_cache_tokens"),
                "server_max_concurrency_x": srv.get("max_concurrency_x"),
            })
            rows.append(row)
            append_summary(outdir, [row])

        docker_rm()
        time.sleep(mx["defaults"]["teardown_pause_s"])

    print(f"\nDONE. Results in {outdir}")
    print(f"Report:  python3 {Path(__file__).resolve().parent / 'report.py'} {outdir}")


if __name__ == "__main__":
    main()
