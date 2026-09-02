# Qwen3.8-27B-Uncensored-FP8 on DGX Spark (GB10) — benchmark report

Result dir: `models/deepseek-coder-v2-lite-fp8/bench/results/overnight_20260902_050212`  ·  31 successful runs

Served via `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0), `VLLM_USE_DEEP_GEMM=0`, FP8 block-quant weights, MTP draft head.

**Peak output throughput observed:** 172.7 tok/s (`kv_auto` / `batch8_balanced`, conc 8).


### All runs

| config | workload | model | spec tok | conc | out tok/s | total tok/s | ttft p50 | tpot | accept len | mem peak MB | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| backend_auto | single_stream | fp8 | 0 | 1 | 68.6 | 103.3 | 104 | 14.5 | 0.00 | 44467 | 95 |
| backend_auto | batch8_balanced | fp8 | 0 | 8 | 171.7 | 344.4 | 633 | 46.0 | 0.00 | 44552 | 132 |
| backend_triton | single_stream | fp8 | 0 | 1 | 68.9 | 103.7 | 106 | 14.4 | 0.00 | 44236 | 95 |
| backend_triton | batch8_balanced | fp8 | 0 | 8 | 168.2 | 337.5 | 760 | 46.9 | 0.00 | 44253 | 134 |
| kv_auto | single_stream | fp8 | 0 | 1 | 69.0 | 103.9 | 104 | 14.4 | 0.00 | 44540 | 94 |
| kv_auto | batch8_balanced | fp8 | 0 | 8 | 172.7 | 346.4 | 604 | 45.8 | 0.00 | 44556 | 130 |
| kv_auto | codegen_decode | fp8 | 0 | 8 | 166.4 | 208.3 | 695 | 47.9 | 0.00 | 44526 | 325 |
| kv_auto | ctx65k_load | fp8 | 0 | 4 | 74.4 | 595.1 | 8591 | 41.8 | 0.00 | 45971 | 509 |
| seqs_1 | single_stream | fp8 | 0 | 1 | 68.9 | 103.8 | 103 | 14.4 | 0.00 | 44214 | 95 |
| seqs_1 | batch4_balanced | fp8 | 0 | 4 | 68.2 | 136.8 | 45191 | 14.6 | 0.00 | 44151 | 231 |
| seqs_1 | codegen_decode | fp8 | 0 | 8 | 67.6 | 84.6 | 212286 | 14.8 | 0.00 | 44139 | 672 |
| seqs_2 | single_stream | fp8 | 0 | 1 | 69.0 | 103.9 | 104 | 14.4 | 0.00 | 44636 | 95 |
| seqs_2 | batch4_balanced | fp8 | 0 | 4 | 98.5 | 197.6 | 20981 | 20.1 | 0.00 | 44626 | 166 |
| seqs_2 | codegen_decode | fp8 | 0 | 8 | 97.5 | 122.0 | 126309 | 20.5 | 0.00 | 44589 | 487 |
| seqs_4 | single_stream | fp8 | 0 | 1 | 69.0 | 104.0 | 105 | 14.4 | 0.00 | 44368 | 94 |
| seqs_4 | batch4_balanced | fp8 | 0 | 4 | 131.2 | 263.2 | 420 | 30.1 | 0.00 | 44383 | 131 |
| seqs_4 | codegen_decode | fp8 | 0 | 8 | 128.4 | 160.7 | 64239 | 31.1 | 0.00 | 44349 | 385 |
| seqs_8 | single_stream | fp8 | 0 | 1 | 68.7 | 103.5 | 105 | 14.5 | 0.00 | 44503 | 95 |
| seqs_8 | batch4_balanced | fp8 | 0 | 4 | 133.2 | 267.2 | 351 | 29.7 | 0.00 | 44490 | 128 |
| seqs_8 | codegen_decode | fp8 | 0 | 8 | 166.4 | 208.3 | 688 | 47.9 | 0.00 | 44495 | 326 |
| seqs_16 | single_stream | fp8 | 0 | 1 | 68.7 | 103.5 | 106 | 14.5 | 0.00 | 44446 | 95 |
| seqs_16 | batch4_balanced | fp8 | 0 | 4 | 132.4 | 265.6 | 389 | 29.8 | 0.00 | 44523 | 130 |
| seqs_16 | codegen_decode | fp8 | 0 | 8 | 166.8 | 208.7 | 686 | 47.8 | 0.00 | 44426 | 326 |
| prefix_off | agent_turn_solo | fp8 | 0 | 1 | 57.6 | 891.7 | 1109 | 15.5 | 0.00 | 44591 | 236 |
| prefix_off | agent_turn_parallel | fp8 | 0 | 8 | 119.5 | 1750.6 | 1254 | 61.0 | 0.00 | 44598 | 236 |
| prefix_on | agent_turn_solo | fp8 | 0 | 1 | 64.0 | 991.6 | 128 | 15.4 | 0.00 | 44485 | 214 |
| prefix_on | agent_turn_parallel | fp8 | 0 | 8 | 159.8 | 2341.0 | 214 | 47.2 | 0.00 | 45148 | 181 |
| ctx_8k | ctx8k_load | fp8 | 0 | 8 | 153.8 | 846.9 | 1122 | 40.4 | 0.00 | 44543 | 93 |
| ctx_32k | ctx32k_load | fp8 | 0 | 6 | 98.6 | 558.8 | 4973 | 47.2 | 0.00 | 44775 | 359 |
| ctx_65k | ctx65k_load | fp8 | 0 | 4 | 74.4 | 595.2 | 8549 | 41.8 | 0.00 | 46709 | 509 |
| ctx_131k | ctx131k_load | fp8 | 0 | 2 | 47.0 | 376.4 | 15702 | 32.2 | 0.00 | 52243 | 1028 |
