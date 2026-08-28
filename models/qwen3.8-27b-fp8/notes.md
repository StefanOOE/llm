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

## Sources

- Model card: <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8>
- Benchmark report: `bench/results/overnight_20260827_191752/report.md`
  (and `report.html` — also at
  <https://claude.ai/code/artifact/0843d0cb-f656-4f09-8e08-3a6fa499fd67>)
