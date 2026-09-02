# Qwen3.8-27B-Uncensored-FP8 on DGX Spark (GB10) — benchmark report

Result dir: `models/qwen3-coder-30b-a3b-fp8/bench/results/overnight_20260902_145242`  ·  4 successful runs

Served via `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0), `VLLM_USE_DEEP_GEMM=0`, FP8 block-quant weights, MTP draft head.

**Peak output throughput observed:** 154.7 tok/s (`backend_auto` / `batch8_balanced`, conc 8).


### All runs

| config | workload | model | spec tok | conc | out tok/s | total tok/s | ttft p50 | tpot | accept len | mem peak MB | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backend_auto | single_stream | fp8 | 0 | 1 | 56.2 | 84.7 | 149 | 17.7 | 0.00 | 61570 | 115 |
| backend_auto | batch8_balanced | fp8 | 0 | 8 | 154.7 | 310.6 | 746 | 50.9 | 0.00 | 61619 | 151 |
| backend_triton | single_stream | fp8 | 0 | 1 | 56.1 | 84.6 | 148 | 17.7 | 0.00 | 61153 | 115 |
| backend_triton | batch8_balanced | fp8 | 0 | 8 | 154.6 | 310.4 | 745 | 50.9 | 0.00 | 61198 | 151 |
