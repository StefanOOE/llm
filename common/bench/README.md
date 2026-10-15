# Benchmark harness

Container-per-config vLLM sweeps, driven by `vllm bench serve`. Generic over
models: point it at a model directory and it reads that model's
`model.env` + `bench/matrix.yaml`.

```bash
# stop the serving instance first — the benchmark reuses the model's port
models/<slug>/serve stop            # or: sudo systemctl stop llm-vllm@<slug>

sg docker -c ".hf-venv/bin/python common/bench/run.py --model models/<slug> --suite smoke"
sg docker -c ".hf-venv/bin/python common/bench/run.py --model models/<slug> --suite overnight"
python3 common/bench/report.py models/<slug>/bench/results/<suite>_<ts>

# vision vs --language-model-only A/B
sg docker -c ".hf-venv/bin/python common/bench/ab_lmonly.py --model models/<slug>"
```

Flags: `--dry-run` (print plan), `--only a,b` (restrict configs),
`--keep-going` (survive a failed config start),
`--resume <dir>` (skip `(config,workload)` pairs already `ok`).

## Files

| | |
|---|---|
| `run.py` | the sweep runner |
| `report.py` | `<results>/summary.csv` → `report.md` + PNG plots + `report.html` |
| `ab_lmonly.py` | vision-tower A/B for one model |
| `_env.py` | shared: parse `box.env` + `model.env`, resolve HF snapshots |

Results land in `models/<slug>/bench/results/<suite>_<ts>/` — raw per-run JSON,
`server_<config>.log`, incremental `summary.csv` / `summary.json`.

## `matrix.yaml` (per model)

Only the sweep: `models` (repo ids to resolve from the HF cache — a family can
list several, e.g. an FP8 and a BF16 checkpoint for comparison), `common_flags`
+ `env` applied to every `vllm serve`, `config_defaults` / `workload_defaults`,
the `workloads`, the `configs` (per-config overrides), and `suites` (ordered
`[config, [workloads]]` lists; `pairs:` for config↔workload pairing;
`include:` to compose suites).

Image, caches, repo id, served name, and port come from
`common/box.env` + `models/<slug>/model.env`, not this file.

## GB10 specifics baked in

- `VLLM_USE_DEEP_GEMM=0` (asserts on sm_121) — in `box.env` and `matrix.yaml:env`
- `nvidia-smi` reports `N/A` for GPU memory → the harness samples `free -m`
- 128 GB **unified**: `--gpu-memory-utilization` carves from the whole pool

## Metrics per run (`summary.csv`)

throughput (`output_tok_s`, `total_tok_s`, `req_s`, `max_output_tok_s`),
latency (`*_ttft_ms`, `*_tpot_ms`, `mean_itl_ms`, `mean_e2el_ms`),
speculative decoding (`spec_accept_rate`, `spec_accept_len`, `spec_pos0..3`),
server startup profile (`server_kv_cache_gib`, `server_kv_cache_tokens`,
`server_max_concurrency_x`, `server_model_load_s`),
unified memory (`mem_base_mb` → `mem_ready_mb` → `mem_peak_mb`).
