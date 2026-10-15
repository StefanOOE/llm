#!/usr/bin/env python3
"""
A/B: --language-model-only  vs  full multimodal, at a model's tuned serving config.

    sg docker -c ".hf-venv/bin/python3 common/bench/ab_lmonly.py --model models/<slug>"

Uses the model's own model.env (MAX_MODEL_LEN / MTP_TOKENS / GPU_MEM_UTIL /
KV_CACHE_DTYPE / parsers). For each variant measures weight-load GiB + seconds,
KV-cache pool, startup wall, unified-memory footprint (free -m), and text
throughput on three workloads. The multimodal variant also runs an OCR request
against models/<slug>/bench/ocr_test.png and reports latency + fields read.

The model's serving port must be free (stop the running server first).
"""
from __future__ import annotations
import argparse, base64, json, re, subprocess, sys, time
from pathlib import Path

from _env import load_model_env, resolve_snapshot

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="model directory (models/<slug>)")
args = ap.parse_args()

MODEL_DIR = Path(args.model).resolve()
SLUG = MODEL_DIR.name
ENV = load_model_env(MODEL_DIR)
HF_CACHE = ENV.get("HF_CACHE", "/home/ss/models/hf-cache")
VLLM_CACHE = ENV.get("VLLM_CACHE", "/home/ss/models/vllm-cache")
IMAGE = ENV.get("VLLM_IMAGE", "nvcr.io/nvidia/vllm:26.07-py3")
SERVED = ENV.get("SERVED_NAME", SLUG)
PORT = int(ENV.get("PORT", 8000))
NAME = f"ab-{SLUG}"
OCR_IMG = MODEL_DIR / "bench" / "ocr_test.png"
OUT = MODEL_DIR / "bench" / "results" / ("ab_lmonly_" + time.strftime("%Y%m%d_%H%M%S"))

SNAP = resolve_snapshot(HF_CACHE, ENV["MODEL"])
if not SNAP:
    sys.exit(f"weights for {ENV['MODEL']} not found under {HF_CACHE}")

WORKLOADS = [  # name, in, out, n, concurrency
    ("single_stream",   512, 1024, 3,  1),
    ("batch8_balanced", 1024, 1024, 16, 8),
    ("saturate32",      1024, 1024, 64, 32),
]

_mtp = int(ENV.get("MTP_TOKENS", 2))
VARIANTS = {
    "text_only":  ["--language-model-only"],
    "multimodal": ["--limit-mm-per-prompt",
                   f'{{"image":{ENV.get("MM_IMAGE_LIMIT", 1)},"video":{ENV.get("MM_VIDEO_LIMIT", 0)}}}'],
}

COMMON = [
    "--served-model-name", SERVED, "--host", "0.0.0.0", "--port", str(PORT),
    "--trust-remote-code",
    "--max-model-len", str(ENV.get("MAX_MODEL_LEN", 16384)),
    "--max-num-seqs", str(ENV.get("MAX_NUM_SEQS", 32)),
    "--gpu-memory-utilization", str(ENV.get("GPU_MEM_UTIL", 0.65)),
    "--kv-cache-dtype", ENV.get("KV_CACHE_DTYPE", "fp8"),
    "--reasoning-parser", ENV.get("REASONING_PARSER", "qwen3"),
    "--enable-auto-tool-choice", "--tool-call-parser", ENV.get("TOOL_CALL_PARSER", "qwen3_coder"),
] + (["--speculative-config", f'{{"method":"mtp","num_speculative_tokens":{_mtp}}}'] if _mtp > 0 else [])


def sh(c, **k): return subprocess.run(c, text=True, capture_output=True, **k)
def rm(): sh(["docker", "rm", "-f", NAME])
def free_used_mb():
    for ln in sh(["free", "-m"]).stdout.splitlines():
        p = ln.split()
        if p and p[0].rstrip(":").lower() in ("mem", "speicher"):
            return int(p[2])
    return None


LOGPAT = {
    "load_gib": r"Model loading took ([\d.]+) GiB",
    "load_s": r"Model loading took [\d.]+ GiB memory and ([\d.]+) seconds",
    "kv_gib": r"Available KV cache memory: ([\d.]+) GiB",
    "kv_tok": r"GPU KV cache size: ([\d,]+) tokens",
    "max_conc": r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x",
}


def start(variant_flags):
    rm()
    base = free_used_mb()
    argv = [
        "docker", "run", "-d", "--name", NAME, "--gpus", "all", "--ipc=host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", "--shm-size=8g",
        "-p", f"{PORT}:{PORT}",
        "-e", "LANG=C.UTF-8", "-e", "LC_ALL=C.UTF-8",
        "-e", "HF_HOME=/hf-cache", "-e", "HF_HUB_OFFLINE=1",
        "-e", f"VLLM_USE_DEEP_GEMM={ENV.get('VLLM_USE_DEEP_GEMM', '0')}",
        "-v", f"{HF_CACHE}:/hf-cache", "-v", f"{VLLM_CACHE}:/root/.cache/vllm",
        IMAGE, "vllm", "serve", SNAP, *COMMON, *variant_flags,
    ]
    (OUT / "cmd").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cp = sh(argv)
    if cp.returncode:
        print("  docker run failed:", cp.stderr.strip()); return None
    ok = False
    while time.time() - t0 < 1500:
        if not sh(["docker", "ps", "-q", "--filter", f"name={NAME}",
                   "--filter", "status=running"]).stdout.strip():
            break
        if sh(["curl", "-s", "-m", "3", "-o", "/dev/null", "-w", "%{http_code}",
               f"http://localhost:{PORT}/health"]).stdout.strip() == "200":
            ok = True; break
        time.sleep(5)
    wall = time.time() - t0
    lg = sh(["docker", "logs", NAME])
    logs = lg.stdout + lg.stderr
    parsed = {}
    for key, pat in LOGPAT.items():
        mo = re.search(pat, logs)
        parsed[key] = float(mo.group(1).replace(",", "")) if mo else None
    return {
        "ok": ok, "startup_s": round(wall, 1),
        "mem_base": base, "mem_ready": free_used_mb(),
        "log": logs, **parsed,
    }


def bench(name, i, o, n, c):
    rf = f"{name}.json"
    sh(["docker", "exec", NAME, "rm", "-f", f"/hf-cache/bench-tmp/{rf}"])
    cp = sh(["docker", "exec", "-e", "LANG=C.UTF-8", NAME, "vllm", "bench", "serve",
             "--backend", "openai-chat", "--base-url", f"http://localhost:{PORT}",
             "--endpoint", "/v1/chat/completions", "--model", SERVED, "--tokenizer", SNAP,
             "--dataset-name", "random", "--random-input-len", str(i),
             "--random-output-len", str(o), "--num-prompts", str(n),
             "--max-concurrency", str(c), "--request-rate", "inf", "--ignore-eos",
             "--num-warmups", "3", "--seed", "0",
             "--save-result", "--result-dir", "/hf-cache/bench-tmp", "--result-filename", rf],
            timeout=3600)
    raw = sh(["docker", "exec", NAME, "cat", f"/hf-cache/bench-tmp/{rf}"]).stdout
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return {"name": name, "err": cp.stderr[-400:]}
    return {"name": name, "out_tok_s": round(d.get("output_throughput", 0), 1),
            "ttft_p50": round(d.get("median_ttft_ms", 0)),
            "accept_len": round(d.get("spec_decode_acceptance_length") or 0, 2)}


def ocr_test():
    if not OCR_IMG.is_file():
        return {"latency_s": 0, "ok": False, "raw": f"missing {OCR_IMG}"}
    b64 = base64.b64encode(OCR_IMG.read_bytes()).decode()
    payload = {
        "model": SERVED, "max_tokens": 400, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Read every line of text in this image verbatim, "
             "then list the shapes and their colours."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        ]}],
    }
    p = HERE / "_ocr_payload.json"
    p.write_text(json.dumps(payload))
    t0 = time.time()
    r = sh(["curl", "-s", "-m", "180", f"http://localhost:{PORT}/v1/chat/completions",
            "-H", "Content-Type: application/json", "--data", "@" + str(p)])
    dt = time.time() - t0
    p.unlink(missing_ok=True)
    try:
        d = json.loads(r.stdout)
        txt = d["choices"][0]["message"]["content"] or ""
    except Exception:
        return {"latency_s": round(dt, 1), "ok": False, "raw": r.stdout[:400]}
    want = ["2026-0842", "1,337", "2026-09-15", "GB10-QWEN-BENCH"]
    hit = [w for w in want if w.replace(",", "") in txt.replace(",", "")]
    return {"latency_s": round(dt, 1), "ok": len(hit) >= 3, "fields_read": hit,
            "usage": d.get("usage"), "text": txt}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (Path(HF_CACHE) / "bench-tmp").mkdir(exist_ok=True)
    results = {}
    for variant, flags in VARIANTS.items():
        print(f"\n{'='*66}\n{variant}   {' '.join(flags)}\n{'='*66}", flush=True)
        s = start(flags)
        if not s or not s["ok"]:
            print("  server did not come up; see log")
            (OUT / f"server_{variant}.log").write_text(s["log"] if s else "")
            results[variant] = {"start": s, "bench": [], "ocr": None}
            rm(); time.sleep(10); continue
        (OUT / f"server_{variant}.log").write_text(s["log"])
        mm = re.search(r"limits of multimodal.*?mode\.|Maximum concurrency.*", s["log"])
        print(f"  up in {s['startup_s']}s | weights {s['load_gib']} GiB / {s['load_s']}s | "
              f"KV {s['kv_gib']} GiB ({s['kv_tok']} tok, {s['max_conc']}x) | "
              f"free {s['mem_base']}->{s['mem_ready']} MB", flush=True)
        b = [bench(*w) for w in WORKLOADS]
        for r in b:
            print("   ", r, flush=True)
        o = None
        if variant == "multimodal":
            o = ocr_test()
            print("  OCR:", {k: o[k] for k in o if k != "text"}, flush=True)
            if o.get("text"):
                print("  --- model output ---\n" + o["text"][:800] + "\n  ---")
        results[variant] = {"start": {k: s[k] for k in s if k != "log"}, "bench": b, "ocr": o}
        rm(); time.sleep(12)

    (OUT / "ab_results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{'='*66}\nSUMMARY   ->  {OUT}/ab_results.json\n{'='*66}")
    a, m = results.get("text_only", {}).get("start", {}), results.get("multimodal", {}).get("start", {})
    if a and m:
        rows = [
            ("weight load GiB", a.get("load_gib"), m.get("load_gib")),
            ("weight load s", a.get("load_s"), m.get("load_s")),
            ("startup wall s", a.get("startup_s"), m.get("startup_s")),
            ("KV pool GiB", a.get("kv_gib"), m.get("kv_gib")),
            ("KV tokens", a.get("kv_tok"), m.get("kv_tok")),
            ("max concurrency x", a.get("max_conc"), m.get("max_conc")),
            ("mem ready MB", a.get("mem_ready"), m.get("mem_ready")),
            ("mem delta MB", (a.get("mem_ready") or 0) - (a.get("mem_base") or 0),
                             (m.get("mem_ready") or 0) - (m.get("mem_base") or 0)),
        ]
        print(f"  {'metric':20s} {'text_only':>14s} {'multimodal':>14s}  delta")
        for k, x, y in rows:
            try: dd = f"{(float(y)-float(x)):+.1f}"
            except Exception: dd = "-"
            print(f"  {k:20s} {str(x):>14s} {str(y):>14s}  {dd}")
    for w in WORKLOADS:
        ba = next((r for r in results.get("text_only", {}).get("bench", []) if r["name"] == w[0]), {})
        bm = next((r for r in results.get("multimodal", {}).get("bench", []) if r["name"] == w[0]), {})
        print(f"  {w[0]:20s} text {ba.get('out_tok_s')} tok/s   mm {bm.get('out_tok_s')} tok/s")
    oc = results.get("multimodal", {}).get("ocr")
    if oc:
        print(f"  OCR: {'PASS' if oc.get('ok') else 'CHECK'} in {oc.get('latency_s')}s, "
              f"read {oc.get('fields_read')}")


if __name__ == "__main__":
    main()
