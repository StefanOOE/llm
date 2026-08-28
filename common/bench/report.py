#!/usr/bin/env python3
"""
Turn a benchmark result directory (produced by run.py) into report.md + PNG plots.

    python3 common/bench/report.py models/<slug>/bench/results/<suite>_<ts>
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load(outdir: Path) -> list[dict]:
    rows = list(csv.DictReader((outdir / "summary.csv").open()))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("config", "workload", "status", "model", "spec", "kv_cache_dtype",
                     "prefix_caching"):
                continue
            r[k] = num(v)
    return [r for r in rows if r.get("status") == "ok"]


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def section_mtp(rows, outdir, lines):
    mtp = [r for r in rows if r["config"].endswith(("spec_off", "spec_1", "spec_2",
                                                    "spec_3", "spec_4"))]
    if not mtp:
        return
    for model in sorted({r["model"] for r in mtp}):
        mm = [r for r in mtp if r["model"] == model]
        lines.append(f"\n### MTP speculative-token sweep — {model.upper()}\n")
        wls = sorted({r["workload"] for r in mm})
        for wl in wls:
            ww = sorted((r for r in mm if r["workload"] == wl),
                        key=lambda r: r["num_spec_tokens"])
            lines.append(f"\n**{wl}**  (in={int(ww[0]['input_len'])} "
                         f"out={int(ww[0]['output_len'])} conc={int(ww[0]['concurrency'])})\n")
            base = next((r["output_tok_s"] for r in ww if r["num_spec_tokens"] == 0), None)
            tbl = []
            for r in ww:
                spd = (r["output_tok_s"] / base) if base else None
                tbl.append([
                    int(r["num_spec_tokens"]),
                    fmt(r["output_tok_s"], 1),
                    fmt(spd, 2) + "x" if spd else "",
                    fmt(r["spec_accept_len"], 2),
                    fmt(r["spec_accept_rate"], 1) if r["spec_accept_rate"] else "",
                    fmt(r["p50_ttft_ms"], 0),
                    fmt(r["mean_tpot_ms"], 1),
                    fmt(r["mem_peak_mb"], 0),
                ])
            lines.append(md_table(
                ["spec tok", "out tok/s", "vs off", "accept len", "accept %",
                 "ttft p50 ms", "tpot ms", "mem peak MB"], tbl))

        if HAVE_MPL:
            fig, ax = plt.subplots(figsize=(8, 5))
            for wl in wls:
                ww = sorted((r for r in mm if r["workload"] == wl),
                            key=lambda r: r["num_spec_tokens"])
                ax.plot([r["num_spec_tokens"] for r in ww],
                        [r["output_tok_s"] for r in ww], marker="o", label=wl)
            ax.set_xlabel("num_speculative_tokens (0 = MTP off)")
            ax.set_ylabel("output throughput (tok/s)")
            ax.set_title(f"MTP sweep — {model.upper()}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            p = outdir / f"mtp_sweep_{model}.png"
            fig.tight_layout()
            fig.savefig(p, dpi=140)
            plt.close(fig)
            lines.append(f"\n![mtp sweep {model}]({p.name})\n")


def section_context(rows, outdir, lines):
    ctx = [r for r in rows if "_ctx_" in r["config"]]
    if not ctx:
        return
    ctx.sort(key=lambda r: r["max_model_len"])
    lines.append("\n### Context-length scaling — FP8, MTP=3\n")
    tbl = []
    for r in ctx:
        tbl.append([
            f"{int(r['max_model_len'])//1024}k",
            int(r["input_len"]), int(r["output_len"]), int(r["concurrency"]),
            fmt(r["output_tok_s"], 1),
            fmt(r["p50_ttft_ms"], 0),
            fmt(r["mean_tpot_ms"], 1),
            fmt(r["spec_accept_len"], 2),
            fmt(r["server_kv_cache_gib"], 1),
            f"{int(r['server_kv_cache_tokens'])//1000}k" if r["server_kv_cache_tokens"] else "",
            fmt(r["server_max_concurrency_x"], 1),
            fmt(r["server_model_load_s"], 0),
            fmt(r["mem_peak_mb"], 0),
        ])
    lines.append(md_table(
        ["ctx", "in", "out", "conc", "out tok/s", "ttft p50 ms", "tpot ms",
         "accept len", "KV GiB", "KV tokens", "max conc x", "load s", "mem peak MB"], tbl))

    if HAVE_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        xs = [r["max_model_len"] / 1024 for r in ctx]
        axes[0].plot(xs, [r["output_tok_s"] for r in ctx], marker="o")
        axes[0].set_title("output tok/s vs context"); axes[0].set_xlabel("context (k)")
        axes[1].plot(xs, [r["p50_ttft_ms"] for r in ctx], marker="o", color="tab:red")
        axes[1].set_title("TTFT p50 (ms) vs context"); axes[1].set_xlabel("context (k)")
        axes[2].plot(xs, [r["server_kv_cache_gib"] for r in ctx], marker="o", color="tab:green")
        axes[2].set_title("KV cache pool (GiB) vs context"); axes[2].set_xlabel("context (k)")
        for a in axes:
            a.grid(True, alpha=0.3)
            a.set_xscale("log", base=2)
        fig.tight_layout()
        p = outdir / "context_scaling_fp8.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        lines.append(f"\n![context scaling]({p.name})\n")


def section_fp8_vs_bf16(rows, outdir, lines):
    have = {r["model"] for r in rows}
    if not {"fp8", "bf16"} <= have:
        return
    mtp = [r for r in rows if r["workload"] in
           ("single_stream", "batch8_balanced", "codegen_decode", "saturate32")]
    wls = ["single_stream", "batch8_balanced", "codegen_decode", "saturate32"]

    lines.append("\n### FP8 vs BF16 — best MTP setting per workload\n")
    tbl, plot = [], []
    for wl in wls:
        f = [r for r in mtp if r["model"] == "fp8" and r["workload"] == wl]
        b = [r for r in mtp if r["model"] == "bf16" and r["workload"] == wl]
        if not (f and b):
            continue
        fb = max(f, key=lambda r: r["output_tok_s"])
        bb = max(b, key=lambda r: r["output_tok_s"])
        winner = "FP8" if fb["output_tok_s"] >= bb["output_tok_s"] else "BF16"
        ratio = fb["output_tok_s"] / bb["output_tok_s"]
        tbl.append([wl, int(fb["concurrency"]),
                    f"{fb['output_tok_s']:.1f} (mtp{int(fb['num_spec_tokens'])})",
                    f"{bb['output_tok_s']:.1f} (mtp{int(bb['num_spec_tokens'])})",
                    f"{ratio:.2f}x", winner])
        plot.append((wl, fb["output_tok_s"], bb["output_tok_s"]))
    lines.append(md_table(
        ["workload", "conc", "FP8 best", "BF16 best", "FP8/BF16", "winner"], tbl))

    lines.append("\n**Matched MTP token counts** (out tok/s):\n")
    keys = sorted({r["num_spec_tokens"] for r in mtp})
    tbl2 = []
    for wl in wls:
        for st in keys:
            f = next((r for r in mtp if r["model"] == "fp8" and r["workload"] == wl
                      and r["num_spec_tokens"] == st), None)
            b = next((r for r in mtp if r["model"] == "bf16" and r["workload"] == wl
                      and r["num_spec_tokens"] == st), None)
            if not (f and b):
                continue
            tbl2.append([wl, int(st), fmt(f["output_tok_s"], 1), fmt(b["output_tok_s"], 1),
                         fmt(f["output_tok_s"] / b["output_tok_s"], 2) + "x"])
    lines.append(md_table(["workload", "spec tok", "fp8 tok/s", "bf16 tok/s", "fp8/bf16"], tbl2))

    if HAVE_MPL and plot:
        import numpy as np
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(plot))
        ax.bar(x - 0.2, [p[1] for p in plot], 0.4, label="FP8 (best MTP)")
        ax.bar(x + 0.2, [p[2] for p in plot], 0.4, label="BF16 (best MTP)")
        ax.set_xticks(x)
        ax.set_xticklabels([p[0] for p in plot], rotation=15, ha="right")
        ax.set_ylabel("output throughput (tok/s)")
        ax.set_title("FP8 vs BF16 — best MTP setting per workload (GB10)")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = outdir / "fp8_vs_bf16.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        lines.append(f"\n![fp8 vs bf16]({p.name})\n")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    outdir = Path(sys.argv[1])
    rows = load(outdir)
    if not rows:
        sys.exit("no successful rows in summary.csv")

    lines = [f"# Qwen3.8-27B-Uncensored-FP8 on DGX Spark (GB10) — benchmark report",
             f"\nResult dir: `{outdir}`  ·  {len(rows)} successful runs",
             "\nServed via `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0), "
             "`VLLM_USE_DEEP_GEMM=0`, FP8 block-quant weights, MTP draft head.\n"]

    peak = max((r["output_tok_s"] for r in rows), default=0)
    best = next(r for r in rows if r["output_tok_s"] == peak)
    lines.append(f"**Peak output throughput observed:** {peak:.1f} tok/s "
                 f"(`{best['config']}` / `{best['workload']}`, "
                 f"conc {int(best['concurrency'])}).\n")

    section_mtp(rows, outdir, lines)
    section_context(rows, outdir, lines)
    section_fp8_vs_bf16(rows, outdir, lines)

    lines.append("\n### All runs\n")
    hdr = ["config", "workload", "model", "spec tok", "conc", "out tok/s",
           "total tok/s", "ttft p50", "tpot", "accept len", "mem peak MB", "wall s"]
    tbl = [[r["config"], r["workload"], r["model"], int(r["num_spec_tokens"]),
            int(r["concurrency"]), fmt(r["output_tok_s"], 1), fmt(r["total_tok_s"], 1),
            fmt(r["p50_ttft_ms"], 0), fmt(r["mean_tpot_ms"], 1),
            fmt(r["spec_accept_len"], 2), fmt(r["mem_peak_mb"], 0), fmt(r["wall_s"], 0)]
           for r in rows]
    lines.append(md_table(hdr, tbl))

    (outdir / "report.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {outdir/'report.md'}"
          + ("" if HAVE_MPL else "  (matplotlib missing: no plots)"))


if __name__ == "__main__":
    main()
