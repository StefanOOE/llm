# Qwen3.8-27B-Uncensored-FP8 on DGX Spark (GB10) — benchmark report

Result dir: `models/qwen3-coder-30b-a3b-fp8/bench/results/overnight_20260902_153547`  ·  35 successful runs

Served via `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0), `VLLM_USE_DEEP_GEMM=0`, FP8 block-quant weights, MTP draft head.

**Peak output throughput observed:** 301.9 tok/s (`seqs_32` / `saturate32`, conc 32).


### All runs

| config | workload | model | spec tok | conc | out tok/s | total tok/s | ttft p50 | tpot | accept len | mem peak MB | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backend_auto | single_stream | fp8 | 0 | 1 | 55.9 | 84.3 | 148 | 17.8 | 0.00 | 71839 | 116 |
| backend_auto | batch8_balanced | fp8 | 0 | 8 | 154.5 | 310.3 | 763 | 50.9 | 0.00 | 72320 | 152 |
| backend_triton | single_stream | fp8 | 0 | 1 | 56.0 | 84.4 | 148 | 17.7 | 0.00 | 72101 | 116 |
| backend_triton | batch8_balanced | fp8 | 0 | 8 | 151.6 | 304.3 | 748 | 52.0 | 0.00 | 72081 | 154 |
| kv_auto | single_stream | fp8 | 0 | 1 | 55.0 | 82.9 | 148 | 18.1 | 0.00 | 72132 | 118 |
| kv_auto | batch8_balanced | fp8 | 0 | 8 | 140.2 | 281.4 | 765 | 56.2 | 0.00 | 71956 | 163 |
| kv_auto | codegen_decode | fp8 | 0 | 8 | 123.8 | 155.0 | 870 | 64.4 | 0.00 | 71964 | 446 |
| kv_fp8 | single_stream | fp8 | 0 | 1 | 56.1 | 84.6 | 151 | 17.7 | 0.00 | 72461 | 115 |
| kv_fp8 | batch8_balanced | fp8 | 0 | 8 | 155.3 | 311.8 | 752 | 50.8 | 0.00 | 72516 | 151 |
| kv_fp8 | codegen_decode | fp8 | 0 | 8 | 134.7 | 168.7 | 831 | 59.2 | 0.00 | 72519 | 411 |
| seqs_1 | single_stream | fp8 | 0 | 1 | 55.9 | 84.3 | 150 | 17.8 | 0.00 | 71827 | 116 |
| seqs_1 | batch4_balanced | fp8 | 0 | 4 | 55.2 | 110.9 | 55810 | 17.9 | 0.00 | 71791 | 284 |
| seqs_4 | single_stream | fp8 | 0 | 1 | 55.9 | 84.3 | 152 | 17.8 | 0.00 | 71953 | 116 |
| seqs_4 | batch4_balanced | fp8 | 0 | 4 | 101.0 | 202.7 | 533 | 39.1 | 0.00 | 72009 | 168 |
| seqs_4 | saturate16 | fp8 | 0 | 16 | 102.0 | 204.9 | 120664 | 38.9 | 0.00 | 71976 | 367 |
| seqs_8 | single_stream | fp8 | 0 | 1 | 55.9 | 84.2 | 147 | 17.8 | 0.00 | 72050 | 116 |
| seqs_8 | batch8_balanced | fp8 | 0 | 8 | 152.9 | 307.1 | 761 | 51.5 | 0.00 | 72126 | 153 |
| seqs_8 | saturate16 | fp8 | 0 | 16 | 149.9 | 301.0 | 54682 | 52.8 | 0.00 | 72103 | 264 |
| seqs_16 | single_stream | fp8 | 0 | 1 | 55.8 | 84.2 | 149 | 17.8 | 0.00 | 72499 | 116 |
| seqs_16 | batch8_balanced | fp8 | 0 | 8 | 151.0 | 303.3 | 756 | 52.1 | 0.00 | 72469 | 154 |
| seqs_16 | saturate32 | fp8 | 0 | 32 | 213.7 | 429.1 | 75897 | 74.1 | 0.00 | 72472 | 352 |
| seqs_32 | single_stream | fp8 | 0 | 1 | 55.8 | 84.2 | 150 | 17.8 | 0.00 | 71730 | 116 |
| seqs_32 | batch8_balanced | fp8 | 0 | 8 | 153.6 | 308.3 | 759 | 51.3 | 0.00 | 71689 | 152 |
| seqs_32 | saturate32 | fp8 | 0 | 32 | 301.9 | 606.1 | 1117 | 104.2 | 0.00 | 72073 | 263 |
| prefix_off | agent_turn_solo | fp8 | 0 | 1 | 45.7 | 708.6 | 1441 | 19.4 | 0.00 | 72247 | 295 |
| prefix_off | agent_turn_parallel | fp8 | 0 | 8 | 101.4 | 1485.9 | 1675 | 71.2 | 0.00 | 72225 | 278 |
| prefix_on | agent_turn_solo | fp8 | 0 | 1 | 50.8 | 787.1 | 180 | 19.4 | 0.00 | 72203 | 268 |
| prefix_on | agent_turn_parallel | fp8 | 0 | 8 | 148.0 | 2168.7 | 288 | 50.3 | 0.00 | 72117 | 197 |
| ctx_8k | ctx8k_load | fp8 | 0 | 8 | 116.8 | 643.7 | 1612 | 52.2 | 0.00 | 59138 | 121 |
| ctx_32k | ctx32k_load | fp8 | 0 | 4 | 69.0 | 391.0 | 4628 | 45.9 | 0.00 | 59177 | 416 |
| ctx_65k | ctx65k_load | fp8 | 0 | 3 | 45.5 | 363.9 | 11377 | 50.6 | 0.00 | 65396 | 604 |
| ctx_131k | ctx131k_load | fp8 | 0 | 2 | 36.7 | 293.7 | 33305 | 50.3 | 0.00 | 72501 | 1139 |
| ctx_262k | ctx262k_load | fp8 | 0 | 1 | 19.1 | 293.1 | 80439 | 42.4 | 0.00 | 77509 | 1685 |
| batched_4096 | codegen_decode | fp8 | 0 | 8 | 134.7 | 168.6 | 1185 | 59.1 | 0.00 | 71804 | 416 |
| batched_8192 | codegen_decode | fp8 | 0 | 8 | 131.1 | 164.2 | 890 | 60.8 | 0.00 | 71909 | 421 |
