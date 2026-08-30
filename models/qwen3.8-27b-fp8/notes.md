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

## Sources

- Model card: <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8>
- Benchmark report: `bench/results/overnight_20260827_191752/report.md`
  (and `report.html` — also at
  <https://claude.ai/code/artifact/0843d0cb-f656-4f09-8e08-3a6fa499fd67>)
