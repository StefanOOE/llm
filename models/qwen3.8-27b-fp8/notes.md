# orcarouter/Qwen3.8-27B-Uncensored-FP8 — notes

## What it is

- **Base:** `Qwen/Qwen3.8-27B` — abliterated (refusal direction removed), then
  offline block-FP8 quantized to the exact scheme of `Qwen/Qwen3.8-27B-FP8`.
- **Architecture:** `Qwen3_5ForConditionalGeneration` (`qwen3_5`) — 64 layers,
  hidden 5120, hybrid attention (16 full-attention + 48 Gated-DeltaNet linear,
  interval 4), native vision + video tower, MTP speculative-decoding head.
- **Weights:** block-FP8 E4M3, `weight_block_size [128,128]`, ~28 GiB / 7 shards.
  Vision tower, norms, router, embeddings, `lm_head` stay BF16.
- **Context:** 262,144 native.
- **License:** Apache 2.0. **Uncensored** — no built-in guardrails, research use only.

## vLLM specifics

- **`transformers` in the NGC image (5.6.1) is fine** — vLLM 0.24.0 has native
  `qwen3_5` support and does not use HF's modeling code here.
- **`VLLM_USE_DEEP_GEMM=0` is mandatory** on sm_121 (GB10). DeepGEMM's UE8M0 FP8
  path asserts `Unknown recipe` during kernel warmup. vLLM falls back to CUTLASS
  block-scaled FP8. (Set in `common/box.env`.)
- **Do NOT pass `--quantization fp8`** — it is read from the checkpoint's
  `config.json`; setting it explicitly breaks loading.
- **MTP flag** is `--speculative-config '{"method":"mtp","num_speculative_tokens":N}'`.
  The old `--num-speculative-tokens` was removed in 0.24.0.
- **FP8 KV cache** runs with scaling factor 1.0 (no `k_scale`/`v_scale` in the
  checkpoint). Possible minor accuracy impact; not measured here.
- **Parsers** (from the model card): `--reasoning-parser qwen3`,
  `--tool-call-parser qwen3_coder`. Thinking is ON by default; disable per
  request with `chat_template_kwargs={"enable_thinking": false}`. The reasoning
  trace comes back in the `reasoning` field.
- **Vision:** send `content` as a list with a `text` part and an `image_url`
  part (`data:image/png;base64,...`). Loaded by default here — see the A/B in
  `bench/results/ab_lmonly_*`. `LANGUAGE_MODEL_ONLY=1` skips the tower.

## Related models (not deployable on one GB10)

- `orcarouter/Qwen3.8-27B-Uncensored` — the BF16 source, ~56 GiB. Benchmarked
  for the FP8-vs-BF16 comparison; FP8 wins 1.4–1.9× on this bandwidth-bound box.
- `orcarouter/Qwen3.8-Flash-Next-Uncensored*` — `qwen4_exp`, ~180B MoE, **~186 GB
  in any quant** (a 51B-param BF16 PLE n-gram table dominates). Exceeds the
  128 GB unified pool; needs a day-0 x86 vLLM image and TP ≥ 8.

## 2026-08-30: tool-calling HTTP 500 fix + context bump to 65536

A remote agent needing `tools` + >=64k context hit two independent bugs:

- **Tool-calling always 500'd**: `cannot import name 'normalize_tool_choice'
  from 'xgrammar'`. Root cause: base NGC `26.07-py3` ships `xgrammar==0.2.0`,
  but vLLM 0.24.0's tool-calling path needs `normalize_tool_choice`, added in
  xgrammar 0.2.4. Fixed with a `--no-deps` bump in a tiny derived image
  (`docker/Dockerfile.xgrammar024`, built locally as
  `vllm-qwen:26.07-xgrammar024`) so `transformers`/`torch` stay untouched.
  `model.env` overrides `VLLM_IMAGE` to this tag. **Not registry-pullable** —
  if the local Docker image cache ever loses it (`docker image prune`, fresh
  host), rebuild with the command in the Dockerfile's header comment before
  the next start; `preflight()` in `lib.sh` will otherwise try (and fail) to
  `docker pull` a nonexistent remote tag.
  `--tool-call-parser` was deliberately left at `qwen3_coder` (not switched to
  `hermes` as first suggested) — that's the model card's documented format,
  and the bug was xgrammar's import error, unrelated to parser choice.
  Verified: a `tools`-bearing request now returns a clean `tool_calls` field
  instead of 500.
- **Context raised 16384 -> 65536**: the agent hard-requires >=64000. Native
  `max_position_embeddings` is 262144 (config.json) — no YaRN needed, just
  the flag. Sits between the already-benchmarked 32k/64k rows in
  `model.env`'s context table; `GPU_MEM_UTIL`/`MAX_NUM_SEQS` untouched since
  the 64k row already fit inside the 0.65 util budget.

Both changes live in `model.env` (sourced fresh on every start by the
systemd unit) — no `install-service` re-run was needed, and both survive a
reboot as long as the local Docker image persists (see caveat above).

## 2026-09-01: gap-closing sweep at the real production context

Auditing `model.env` against its own cited data (prompted by "what other
optimizations are there for qwen?") found the 2026-08-30 context bump to
65536 had never actually been load-tested at production settings — the
original 43-run sweep (2026-08-27/28) tuned `MTP_TOKENS`/`MAX_NUM_SEQS` at
`max_model_len=32768`, and its context-scaling rows used bench-only
`gpu_memory_utilization=0.80`/reduced `max_num_seqs`, not production's
`0.65`/`32`. Five suites closed this and three smaller gaps:

- **`prod_mtp_sweep_65k`**: production combo (65536 + 0.65 + 32) validated
  together for the first time — works fine, no issues. Also re-settled
  `MTP_TOKENS` at the correct context: 3 wins single-stream (+11%, the
  deployment-representative metric — one agent, not 32 users), ties on
  batch/saturate, loses ~8% on codegen_decode. Switched default 2 -> 3.
  See `model.env`'s MTP block for the full table.
- **`kv_dequant_cliff_check`**: a GB10-specific report (llama.cpp/GGML KV
  quantization) found KV quant can get *slower* than f16 at deep context on
  this bandwidth-rich unified-memory hardware ("dequantization cliff") —
  worth checking since our own `kv_sweep` never tested deep enough. Result
  at 65536 ctx: fp8 KV **wins decisively** (28.52 vs 22.41 tok/s, +27%; TPOT
  63.2ms vs 96.3ms, -34%). Does not reproduce for vLLM's fp8 KV cache
  (different kernel than GGML's blockwise dequant). No change to
  `KV_CACHE_DTYPE`.
- **`prefix_cache_ab`**: `--enable-prefix-caching` had never been A/B
  tested (model.env asserted it was "off" by default with MTP — untrue,
  live `--help=all` shows the real default is `None`/auto). Tested against
  a workload shaped like the actual deployment (6000-token shared system
  prompt/tool schema + short varying turn, not the harness's usual fully-
  independent random prompts — needed a small additive `run.py` extension,
  `--random-prefix-len`/`--random-range-ratio`). Clear win: TTFT -25% solo
  (conc=1), -27% TTFT **and** +22% throughput under concurrency (conc=8).
  **Not enabled by default yet** — `lib.sh`'s `generate` runner path has no
  dedicated env var for this flag (only reachable via `EXTRA_VLLM_ARGS`
  today); wiring a real `PREFIX_CACHING` knob is a tracked follow-up, not
  bundled into this sweep.
- **`seqs_sweep_ext_65k`**: model.env's own comment said "48-64 also fine;
  >32 not tested." Now tested: throughput keeps climbing through 64 (32:
  202.6 tok/s -> 48: 246.3 -> 64: 285.5 tok/s), no ceiling hit, no
  regression (unlike gpt-oss-120b's own seqs sweep, which *did* regress
  past its sweet spot). The real tradeoff is TTFT (2.4s -> 4.9s), not a
  hard limit. `MAX_NUM_SEQS` stays 32 — this deployment is one agent, not a
  batch workload, so the extra throughput headroom at 48/64 isn't needed.
- **`moe_backend_probe`**: model is confirmed MoE + hybrid attention
  (`qwen3_5`, see above), so `--moe-backend` is a real axis unlike a dense
  model — never set before. Tried `triton` against `auto` (the only
  candidate; `flashinfer_cutlass`/`flashinfer_trtllm` skipped per vLLM
  issue #43507 — CUTLASS FP8 MoE broken on SM_120/SM_121, a hardware-class
  issue; `deep_gemm` skipped, fights the box-wide mandatory
  `VLLM_USE_DEEP_GEMM=0`). Mixed result (triton +9.6% single-stream, -7.7%
  batch8_balanced) — no compelling reason to switch. `triton` was manually
  curl-verified correct (non-null `content`/`reasoning`/`tool_calls`)
  before being set aside, same correctness-gating rule as gpt-oss-120b's
  backend sweep. `EXTRA_VLLM_ARGS` stays empty.

**Operational note**: most of these suites hit the same externally-killed
background wrapper issue documented in `gpt-oss-120b/notes.md` — a
background sweep process gets killed (confirmed not the user, happened
repeatedly across an otherwise-unrelated session) while the underlying
container keeps loading/serving. Every affected suite was salvaged by
letting the orphaned container finish, then either driving the remaining
`vllm bench serve` calls directly via `docker exec` (same host/container
path `run.py` itself uses) or waiting it out in the foreground, and folding
results into `summary.csv` via `run.py`'s own `append_summary()` so they're
indistinguishable from a normal run. No data lost, just extra steps.

Full raw results: `bench/results/{prod_mtp_sweep_65k,kv_dequant_cliff_check,
prefix_cache_ab,seqs_sweep_ext_65k,moe_backend_probe}_2026*/`.

Also evaluated and explicitly **not** pursued this round: CUDA-graph/
`--compilation-config` tuning (no existing signal either direction, large
unexplored surface), KV-cache dtypes beyond fp8/auto like `fp8_e5m2`/
`nvfp4` (the current fp8 choice already has an unmeasured accuracy question
sitting under it — checkpoint has no k/v_scale — stacking a new unverified
precision experiment on that is premature), and `bf16_spec_1` (cosmetic
sweep-symmetry gap only, FP8 already decisively won weights).

A user-flagged idea (llama.cpp as an alternative serving stack, motivated
by the same KV-cache research above) is deliberately **out of scope** here
and tracked as its own future planning round — see the plan file this
sweep was executed from for what was found (the GB10-specific
`croll83/llama.cpp-dgx` fork is itself already superseded by upstream
llama.cpp per its own README/discussion history; `llama-server`'s
`--slot-save-path` disk-persisted KV cache is a real alternative to
`--enable-prefix-caching` worth comparing if that round ever happens).

## 2026-09-01: context raised 65536 -> 131072 for the agent harness

The remote agent harness (hits the vLLM on `:8000` as its
backend) needs more than 65536 tokens for its autonomous runs. Raised
`MAX_MODEL_LEN` 65536 -> 131072 in `model.env`.

- **No YaRN**: `config.json` `text_config.max_position_embeddings` is 262144,
  `rope_scaling` null. 131072 is well under native -> just the flag.
- **Memory-neutral**: `--gpu-memory-utilization 0.65` carves a *fixed* slice
  of the unified pool; `max_model_len` does not size the KV pool. Live logs:
  `Available KV cache memory: 45.14 GiB` / `GPU KV cache size: 1,018,148
  tokens` -- unchanged by this bump. Only the vLLM-estimated max concurrency
  drops: 15.54x @ 65536 -> ~7.8x @ 131072, which is still far more than a
  single agent needs (vLLM starts as long as one full-length request fits:
  131072 << 1,018,148). `GPU_MEM_UTIL` / `MAX_NUM_SEQS` untouched; the
  `bge-m3` co-tenant (0.12) is unaffected.
- **Perf tradeoff accepted**: only matters when the context is actually
  filled -- 131072 sits between the 64k row (22.6 tok/s, 32.9 s TTFT) and
  the 256k row (12.9 tok/s, 117 s) of `model.env`'s context-scaling table.
  Explicitly taken (deep-context speed is not important for autonomous
  agentic tasks).
- Applied by `sudo systemctl restart llm-vllm@qwen3.8-27b-fp8` (`model.env`
  is re-sourced on every start); survives reboot, no `install-service`
  re-run -- same as the 2026-08-30 bump.
- **Client side (not in this repo)**: the agent harness's own context-window
  setting, on the machine running the agent, must also be raised to <=131072
  to actually use the extra room.

## 2026-09-10: crash-loop -- unified-pool oversubscription (OOM -> Xid 43)

**Symptom** (user report): "qwen3.8 keeps crashing." Repeated `EngineCore` death; on
boot, ~2 cold-start attempts in 3 crashed ~2.5 min in, then one succeeded.

**Timeline** (all Sep 10):
| time | event |
|---|---|
| (Sep 09 17:35) | qwen up, healthy |
| 06:50 | EngineCore dies **on a live request** (13 h uptime) -- `CUBLAS_STATUS_INTERNAL_ERROR` on a BF16 GEMM `(5120,96)` in a cached inductor artifact -> device-side assert -> `EngineDeadError` |
| 06:53 | manual `apt-get upgrade` (see "nvidia skew" below) |
| 06:55 | reboot |
| 07:00, 07:07 | cold-start crashes at the **same** GEMM -- and the kernel logs the real cause: `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal` then `NVRM: Xid (PCI:000f:01:00): 43` |
| 07:13 | 3rd attempt succeeds |

**Root cause: the 128 GB unified LPDDR5x pool was oversubscribed.**
- qwen `GPU_MEM_UTIL=0.65` (~83 GB: weights 28.95 GiB + KV pool 45.16 GiB +
  activations) + bge-m3 `0.12` (~15 GB, for a 1 GiB pooling model) =
  **0.77 x 128 ≈ 98.6 GB**.
- Steady state (`free -g`, both healthy): used 94 / **free 1 / available 27**.
- Cold-start needs a *transient* peak above steady state (29 GiB weight load
  racing the 28.75 GiB checkpoint into page cache + profiling/warmup run +
  graph capture). When it exceeds free memory: driver `NV_ERR_NO_MEMORY` ->
  cuBLAS can't allocate its workspace -> `CUBLAS_STATUS_INTERNAL_ERROR` ->
  CUDA context poisoned -> `device-side assert`. **The `(5120,96)` GEMM is
  the deterministic surfacing point of vLLM's warmup path, not the bug.**
- The 06:50 crash is the same failure triggered by *runtime* load (a
  long-context request pushed KV+activations to the ceiling; the failed
  allocation was cuBLAS's, so no kernel OOM line that time).

**Fix** (conservative -- keeps 131072 ctx, ~7x concurrency):
- bge-m3 `GPU_MEM_UTIL` 0.12 -> **0.06** (~7.7 GB; still ~4x its real peak).
- qwen `GPU_MEM_UTIL` 0.65 -> **0.58** (~74 GB; KV pool 45 -> ~37 GiB ->
  ~7.1x concurrency at 131072).
- New budget: **qwen 0.58 + bge-m3 0.06 = 0.64 (~82 GB), ~46 GB for host +
  page cache + cold-start peak.** `free` ~1 -> ~18 GB, `available` ~27 -> ~44.
- Stale `torch_compile_cache` (9.3 GB, spanned a `cuda-toolkit-13-config`
  13.3->13.4 bump) cleared during the controlled restart so qwen recompiled
  clean.

**Ruled out** (so nobody re-chases these): GPU hardware fault -- Xid 43 is an
app-level channel exception, recoverable, not Xid 48/79/94/95; bge-m3 ran
fine throughout. ECC -- GB10 doesn't expose it. `crashkernel` -- `1G-:0M` =
0 MB reserved. kdump. GNOME desktop -- ~400 MB total. unattended-upgrades --
disabled (`APT::Periodic::Unattended-Upgrade "0"`); the drift was the manual
`apt-get upgrade`.

**nvidia package skew** (secondary, from the 06:53 `apt-get upgrade`):
`nvidia-modprobe` / `nvidia-settings` / `libxnvctrl0` were pulled to
**615.71.09** while the driver + kernel module + `libnvidia-compute-580`
stayed **580.173.02** (container forward-compat layer: 610.43.02). Reverted
the trio to `610.57.04-1ubuntu1` and `apt-mark hold`'d them + the 580 driver
packages + `cuda-toolkit-13-config-common` so the next `apt upgrade` can't
drift the NVIDIA stack again. Not proven to be the OOM trigger, but a real
mismatch and a loose end. **Do not `apt upgrade` the NVIDIA packages without
unholding deliberately and re-checking the container still starts.**

**No repo guardrail**: `serve.sh` / `lib.sh` don't sum per-model
`GPU_MEM_UTIL` against 1.0. A `common/lib.sh` preflight check that warns when
the running instances' utils + a host reserve exceed the pool would have
caught this -- noted as possible future work, not done here.

## 2026-09-10 (part 2): the real root cause -- GDN Triton autotuner OOM

Part 1's `GPU_MEM_UTIL` cut stopped the *cold-start* crash-loop but qwen
still died on every agent-harness request (~7k tokens, tools). Same underlying
bug, different trigger.

**Root cause** (clean 2026-09-06 stack -- the 09-10 ones were mangled by a
`UnicodeDecodeError` in interpreter teardown, which is why they looked like a
random cuBLAS `INTERNAL_ERROR` on a `(5120,96)` GEMM):

```
qwen3_next.py:562 forward
 -> qwen_gdn_linear_attn.py  (Qwen3NextGatedDeltaNet)
   -> chunk_gated_delta_rule -> fla/ops/chunk_o.py:173 chunk_fwd_o
     -> Triton kernel -> device-side assert   (INSIDE triton autotuner.run)
```

This is **vLLM #36598 / #33857**. On non-SM90 GPUs
`Qwen3NextGatedDeltaNet._forward_core` returns early during the V1 profiling
run (`attn_metadata is None`), so the GDN Triton kernels are never autotuned
while memory is free. vLLM then fills the pool with KV cache. The **first
real GDN prefill of a new shape** fires the autotuner, which benchmarks 16
configs (num_warps [1,2,4,8] x num_stages [2,3,4,5]) with no memory left ->
OOM / device-assert. Part 1's cold-start OOM was this same autotuner firing
during *warmup*; short requests survive because they take the recurrent GDN
path (no chunk autotune). Synthetic large prompts (tried 4: +/- tools, +/-
prefix-cache hits, up to 11k tok) did NOT reproduce -- only the real agent
request shape did.

**Why we're stuck on the bad path:** vLLM's `_resolve_gdn_prefill_backend`
only enables the FlashInfer / CuteDSL GDN prefill kernels for
`is_device_capability(90)` (Hopper) or `is_device_capability_family(100)`
(datacenter Blackwell). GB10 is capability **12.1 / family 120** -> matches
neither -> forced onto the Triton `forward_native` path (live log: `Using
Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)`).
`--gdn-prefill-backend {flashinfer,cutedsl}` can't override it -- the same
family gate sets `supports_* = False`. TensorRT-LLM hits the identical crash
on DGX Spark (NVIDIA/TensorRT-LLM #12230), so the Triton GDN kernels are
shaky on aarch64/consumer-Blackwell generally -- not a vLLM-only bug, but the
fix is vLLM-side.

**Fix applied (Tier 1, no image rebuild):** `DOCKER_ENV=VLLM_TRITON_FORCE_
FIRST_CONFIG=1` in `model.env` (new `DOCKER_ENV` passthrough in `lib.sh`).
vLLM's `triton_utils/force_first_config.py` patches `Autotuner.run` to launch
only the first valid config -> 1 kernel launch instead of 16 -> no OOM.
Confirm active: `[triton-autotune-disabled] kernel=...` in the engine log.
`ENFORCE_EAGER` reverted to 0 (it was set on a wrong hypothesis; per #36598
it only helps the separate CUDA-graph bug #37431, at ~55% throughput cost).

**Upstream fix** is PR #43047 (prewarm the GDN autotuner during profiling) --
lands in vLLM 0.25+ / NGC 26.08+. Tier 2 if the workaround underperforms:
bump the image (rebuild `docker/Dockerfile.xgrammar024` on the new base).

**Tier 3 / strategic:** community consensus (conselara.dev sm121 notes, TRT-LLM
#12230) is that GDN-heavy models are marginal on sm_121 with current
tooling. If the workaround/image-bump don't give acceptable throughput, this
feeds the deferred llama.cpp evaluation or a lower-GDN-ratio model choice.

## Sources

- Model card: <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8>
- Benchmark report: `bench/results/overnight_20260827_191752/report.md`
  (and `report.html` — also at
  <https://claude.ai/code/artifact/0843d0cb-f656-4f09-8e08-3a6fa499fd67>)
- GDN autotuner OOM (2026-09-10 part 2):
  - vLLM #36598 — Triton autotuner OOM on Qwen3.5/Qwen3-Next GDN (non-SM90)
  - vLLM #33857 — Qwen3-Coder-Next Triton allocator error on DGX Spark (GB10)
  - vLLM #37431 — GDN illegal-instruction under CUDA graphs
  - vLLM #37714 — Qwen3.5-27B on SM120 needs `--max-num-batched-tokens 2096`
  - vLLM PR #43047 — prewarm the GDN autotuner during profiling (real fix)
  - vLLM PR #36647 — `--gdn-prefill-backend` config knob
  - NVIDIA/TensorRT-LLM #12230 — identical GDN autotuner crash on DGX Spark
  - <https://conselara.dev/notes/vllm-dgx-spark-sm121-gotchas/>
