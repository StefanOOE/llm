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

## Sources

- Model card: <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8>
- Benchmark report: `bench/results/overnight_20260827_191752/report.md`
  (and `report.html` — also at
  <https://claude.ai/code/artifact/0843d0cb-f656-4f09-8e08-3a6fa499fd67>)
