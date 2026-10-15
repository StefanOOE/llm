# Qwen3.8-27B-Uncensored-FP8 on DGX Spark (GB10) — benchmark report

Result dir: `models/qwen3.8-27b-nvfp4/bench/results/gap_close_nvfp4_20260914_193918`  ·  26 successful runs

Served via `nvcr.io/nvidia/vllm:26.07-py3` (vLLM 0.24.0), `VLLM_USE_DEEP_GEMM=0`, FP8 block-quant weights, MTP draft head.

**Peak output throughput observed:** 275.6 tok/s (`nvfp4_spec_2` / `saturate32`, conc 32).


### MTP speculative-token sweep — NVFP4


**batch8_balanced**  (in=1024 out=1024 conc=8)

| spec tok | out tok/s | vs off | accept len | accept % | ttft p50 ms | tpot ms | mem peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 84.5 | 1.00x | 0.00 |  | 2377 | 92.4 | 74364 |
| 1 | 109.2 | 1.29x | 1.74 | 73.8 | 1281 | 68.3 | 75360 |
| 2 | 122.9 | 1.45x | 2.23 | 61.6 | 1341 | 58.1 | 75384 |
| 3 | 118.4 | 1.40x | 2.65 | 55.1 | 1335 | 53.5 | 76902 |
| 4 | 111.3 | 1.32x | 3.05 | 51.3 | 1105 | 61.5 | 75305 |

**codegen_decode**  (in=1024 out=4096 conc=8)

| spec tok | out tok/s | vs off | accept len | accept % | ttft p50 ms | tpot ms | mem peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 84.8 | 1.00x | 0.00 |  | 2904 | 93.7 | 76476 |
| 1 | 122.3 | 1.44x | 1.84 | 84.0 | 3057 | 62.6 | 75142 |
| 2 | 139.7 | 1.65x | 2.51 | 75.5 | 3406 | 50.6 | 75223 |
| 3 | 146.4 | 1.73x | 2.90 | 63.2 | 3408 | 48.8 | 76870 |
| 4 | 120.1 | 1.42x | 3.20 | 54.9 | 3381 | 59.1 | 75029 |

**saturate32**  (in=1024 out=1024 conc=32)

| spec tok | out tok/s | vs off | accept len | accept % | ttft p50 ms | tpot ms | mem peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 217.4 | 1.00x | 0.00 |  | 3247 | 141.8 | 76492 |
| 1 | 257.6 | 1.18x | 1.74 | 74.0 | 2864 | 113.1 | 75138 |
| 2 | 275.6 | 1.27x | 2.33 | 66.5 | 1922 | 99.0 | 75279 |
| 3 | 263.4 | 1.21x | 2.62 | 53.9 | 2170 | 102.2 | 77218 |

**single_stream**  (in=512 out=1024 conc=1)

| spec tok | out tok/s | vs off | accept len | accept % | ttft p50 ms | tpot ms | mem peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 12.5 | 1.00x | 0.00 |  | 249 | 80.1 | 74027 |
| 1 | 18.4 | 1.48x | 1.76 | 76.1 | 298 | 54.1 | 74693 |
| 2 | 19.9 | 1.60x | 2.06 | 53.1 | 376 | 50.0 | 74720 |
| 3 | 21.4 | 1.72x | 2.38 | 46.1 | 384 | 46.4 | 76471 |
| 4 | 21.4 | 1.72x | 2.51 | 37.6 | 386 | 46.4 | 74628 |

### All runs

| config | workload | model | spec tok | conc | out tok/s | total tok/s | ttft p50 | tpot | accept len | mem peak MB | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nvfp4_spec_off | single_stream | nvfp4 | 0 | 1 | 12.5 | 19.3 | 249 | 80.1 | 0.00 | 74027 | 499 |
| nvfp4_spec_off | batch8_balanced | nvfp4 | 0 | 8 | 84.5 | 173.4 | 2377 | 92.4 | 0.00 | 74364 | 289 |
| nvfp4_spec_off | codegen_decode | nvfp4 | 0 | 8 | 84.8 | 107.1 | 2904 | 93.7 | 0.00 | 76476 | 746 |
| nvfp4_spec_off | saturate32 | nvfp4 | 0 | 32 | 217.4 | 446.0 | 3247 | 141.8 | 0.00 | 76492 | 397 |
| nvfp4_spec_1 | single_stream | nvfp4 | 1 | 1 | 18.4 | 28.5 | 298 | 54.1 | 1.76 | 74693 | 342 |
| nvfp4_spec_1 | batch8_balanced | nvfp4 | 1 | 8 | 109.2 | 224.0 | 1281 | 68.3 | 1.74 | 75360 | 222 |
| nvfp4_spec_1 | codegen_decode | nvfp4 | 1 | 8 | 122.3 | 154.4 | 3057 | 62.6 | 1.84 | 75142 | 506 |
| nvfp4_spec_1 | saturate32 | nvfp4 | 1 | 32 | 257.6 | 528.3 | 2864 | 113.1 | 1.74 | 75138 | 327 |
| nvfp4_spec_2 | single_stream | nvfp4 | 2 | 1 | 19.9 | 30.9 | 376 | 50.0 | 2.06 | 74720 | 300 |
| nvfp4_spec_2 | batch8_balanced | nvfp4 | 2 | 8 | 122.9 | 252.0 | 1341 | 58.1 | 2.23 | 75384 | 196 |
| nvfp4_spec_2 | codegen_decode | nvfp4 | 2 | 8 | 139.7 | 176.3 | 3406 | 50.6 | 2.51 | 75223 | 408 |
| nvfp4_spec_2 | saturate32 | nvfp4 | 2 | 32 | 275.6 | 565.2 | 1922 | 99.0 | 2.33 | 75279 | 300 |
| nvfp4_spec_3 | single_stream | nvfp4 | 3 | 1 | 21.4 | 33.2 | 384 | 46.4 | 2.38 | 76471 | 269 |
| nvfp4_spec_3 | batch8_balanced | nvfp4 | 3 | 8 | 118.4 | 242.9 | 1335 | 53.5 | 2.65 | 76902 | 194 |
| nvfp4_spec_3 | codegen_decode | nvfp4 | 3 | 8 | 146.4 | 184.9 | 3408 | 48.8 | 2.90 | 76870 | 384 |
| nvfp4_spec_3 | saturate32 | nvfp4 | 3 | 32 | 263.4 | 540.3 | 2170 | 102.2 | 2.62 | 77218 | 304 |
| nvfp4_spec_4 | single_stream | nvfp4 | 4 | 1 | 21.4 | 33.2 | 386 | 46.4 | 2.51 | 74628 | 289 |
| nvfp4_spec_4 | batch8_balanced | nvfp4 | 4 | 8 | 111.3 | 228.4 | 1105 | 61.5 | 3.05 | 75305 | 204 |
| nvfp4_spec_4 | codegen_decode | nvfp4 | 4 | 8 | 120.1 | 151.7 | 3381 | 59.1 | 3.20 | 75029 | 424 |
| nvfp4_65k_kv_fp8 | single_stream | nvfp4 | 3 | 1 | 23.7 | 36.7 | 392 | 41.9 | 2.63 | 90275 | 257 |
| nvfp4_65k_kv_fp8 | batch8_balanced | nvfp4 | 3 | 8 | 128.2 | 263.0 | 1123 | 56.1 | 2.56 | 90872 | 185 |
| nvfp4_65k_kv_fp8 | codegen_decode | nvfp4 | 3 | 8 | 153.2 | 193.4 | 3395 | 48.7 | 2.93 | 90808 | 396 |
| nvfp4_65k_kv_fp8 | saturate32 | nvfp4 | 3 | 32 | 259.1 | 531.3 | 1803 | 102.1 | 2.64 | 91509 | 314 |
| nvfp4_65k_kv_fp8 | ctx64k_load | nvfp4 | 3 | 3 | 40.7 | 326.4 | 22329 | 47.2 | 2.93 | 91546 | 663 |
| nvfp4_65k_kv_auto | single_stream | nvfp4 | 3 | 1 | 23.1 | 35.9 | 387 | 42.9 | 2.51 | 95480 | 280 |
| nvfp4_65k_kv_auto | ctx64k_load | nvfp4 | 3 | 3 | 37.6 | 301.3 | 21561 | 56.4 | 3.09 | 96093 | 710 |
