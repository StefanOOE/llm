# orcarouter/Qwen3.8-27B-Uncensored-NVFP4 — notes

**Status (2026-10-04): smoke-tested, all checks passed.** User asked whether
an uncensored variant of `qwen3.8-27b-nvfp4` (the stock NVFP4 checkpoint
serving `gb10-reasoning`) exists. It does — this is it, from the same
`orcarouter` org that already publishes `qwen3.8-27b-fp8` (the uncensored
FP8 checkpoint already running on this box). Added as a third on-demand
`gb10-reasoning` member. Slug shortened from the original
`qwen3.8-27b-uncensored-nvfp4` to `qwen3.8-27b-uc-nvfp4` same day — the
longer name broke `llmctl`'s 27-char slug column alignment.

Gate check: download via `hf download orcarouter/Qwen3.8-27B-Uncensored-NVFP4`
with this box's `HF_TOKEN` succeeded immediately — the account already had
access (confirms the pre-download API guess in the "What it is" section
below).

## Smoke test results (2026-10-04)

Ran ad-hoc (`./serve start`, not installed as a systemd service — this is
still just an evaluation candidate, not adopted). `qwen3.8-27b-nvfp4` was
stopped first (shared port 8000 + memory budget) and restarted afterward to
restore the box to its boot-default state.

- **Clean start.** No device-side assert, no OOM, at the inherited
  conservative `GPU_MEM_UTIL=0.52`.
- **Quant kernel selection resolved correctly despite the different
  encoding** (open question #4 from below, now answered): `auto` picked
  `CutlassFP8ScaledMMLinearKernel` for the FP8 `linear_attn` layers and
  `FlashInferCutlassNvFp4LinearKernel` for the NVFP4 MLP GEMM — **the same
  NVFP4 kernel** `qwen3.8-27b-nvfp4`'s ModelOpt-encoded checkpoint resolved
  to. The compressed-tensors vs. ModelOpt encoding difference flagged as a
  risk in `model.env` turned out not to matter for kernel selection.
- **GDN Triton autotuner workaround confirmed active**: `[triton-autotune-
  disabled]` fired for all the expected kernel variants
  (`chunk_local_cumsum_scalar_kernel`, `chunk_scaled_dot_kkt_fwd_kernel`,
  `merge_16x16_to_64x64_inverse_kernel`, `recompute_w_u_fwd_kernel`,
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`) —
  same as both other qwen3.8-27b checkpoints on this box.
- **KV cache**: 35.03 GiB available, 3.77x max concurrency at 262,144
  tokens/request — slightly less than `qwen3.8-27b-nvfp4`'s 37.57 GiB/4.04x
  at the same context length and `GPU_MEM_UTIL`, consistent with this
  checkpoint's slightly larger weights (~23.0 vs ~21.9 GiB).
- **Functional checks — all pass:**
  - `/v1/models`: correct identity (`gb10-reasoning`, root
    `orcarouter/Qwen3.8-27B-Uncensored-NVFP4`).
  - Plain chat (thinking on and off via `chat_template_kwargs.enable_
    thinking`): correct, coherent, `reasoning` field populated as expected
    with thinking on.
  - Tool calling (`qwen3_coder` parser): correct `tool_calls` structure
    (`get_weather` example), chat template compatibility confirmed.
  - Vision/OCR (`qwen3.8-27b-fp8/bench/ocr_test.png`): read the test invoice
    image correctly, same as the other two checkpoints.
  - **Tokenizer truncation pitfall: refuted.** 30,019-token prompt echoed
    back exactly in `usage.prompt_tokens` — no silent truncation.
  - **Uncensored behavior confirmed**: a lock-picking mechanics question
    (mild refusal-boundary test) answered directly and in detail, no
    refusal/hedging — matches the whole point of picking this checkpoint
    over the stock `qwen3.8-27b-nvfp4`.
- **Rough throughput** (2 casual single-stream samples via
  `common/bench/llm_benchmark.py`, thinking ON by default so these include
  full reasoning-token counts, NOT directly comparable to a fixed-output-
  length benchmark):
  - Run 1: 5,264 tokens (incl. reasoning) / 359.8s ≈ **14.6 tok/s** — very
    long reasoning chain, also likely paid a first-real-request cudagraph-
    capture cost for a new sequence-length bucket (289s to first visible
    token is not representative of steady-state latency).
  - Run 2: 733 tokens / 42.2s ≈ **17.5 tok/s**, 11.8s TTFT.
  - In the same rough ballpark as `qwen3.8-27b-nvfp4`'s own smoke-test
    numbers (18-26 tok/s) but on the lower end — noisy 2-sample comparison
    with a different methodology (variable-length reasoning output, not a
    fixed `max_tokens` cutoff), not a formal head-to-head. A real
    `bench/matrix.yaml` sweep would be needed before drawing a real
    conclusion here, same caveat `qwen3.8-27b-nvfp4`'s own smoke test noted.

**Recommendation**: works correctly end-to-end and delivers the uncensored
behavior it was added for. Still only on-demand, not boot-enabled, not
adopted as any identity's default — that's a separate decision, and would
want a real bench sweep first given the throughput numbers above are too
noisy to act on.

## Production crash 2026-10-10 06:30 (not OOM)

Engine died with `CUDA error: device-side assert triggered` (kernel: NVRM
Xid 43 = GPU stopped the faulting context; no OOM-killer, KV usage 11%).
Stack: `qwen3_next.py` -> `qwen_gdn_linear_attn.py:1570` `_forward_core`,
step 3 "Merge core attention output" (`merged_out.index_copy_` of the spec
and non-spec GDN outputs). Batch at that moment: 2 requests in MTP decode
(3 spec tokens, 40.7k and 1.5k context) + 1 chunked prefill of 1068 tokens
(16.4k context) -- i.e. a mixed spec-decode + prefill batch in the GDN
layer; the index tensors don't match the batch. Looks like a vLLM bug in
GDN + MTP with mixed batches, not this box's memory. systemd restarted it
(healthy again after ~4 min); the in-flight requests got HTTP 500.
Lever if it recurs: `MTP_TOKENS=0` (costs decode speed). First occurrence
for this model; the 2026-10-03 asserts were qwen3.8-27b-nvfp4 at setup.

## What it is

- **Publisher:** orcarouter. **Gated** HF repo — the account behind
  `HF_TOKEN` needs to have clicked "Agree and access repository" for this
  specific repo ID (separate gate from `qwen3.8-27b-fp8`'s, even though same
  org/license). A live API check during research (file listing + full
  `config.json` read via the connected HF account) went through without
  hitting a gate wall — suggests access is likely already granted, but not
  confirmed until a real `hf download` is run on this box.
- **Base model:** `Qwen/Qwen3.8-27B`, uncensored/abliterated derivative (same
  lineage as `qwen3.8-27b-fp8`, NOT the stock release `qwen3.8-27b-nvfp4`
  uses).
- **Architecture:** `Qwen3_5ForConditionalGeneration` — same shape as the
  other two: 64 layers, hybrid attention (16 full + 48 Gated-DeltaNet
  linear, interval 4), native vision + video tower, MTP head
  (`mtp_num_hidden_layers: 1`).
- **Quant scheme — mixed NVFP4/FP8, verified from this checkpoint's own
  `config.json` `quantization_config`** (compressed-tensors format, NOT
  ModelOpt — see below):
  - `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (GDN attention layers) →
    **FP8** (`format: float-quantized`, 8-bit, per-channel, symmetric)
  - `mlp.{gate,up,down}_proj` → **NVFP4** (`format: nvfp4-pack-quantized`,
    4-bit, `group_size: 16`, tensor-group strategy)
  - Same split as `qwen3.8-27b-nvfp4` (attention stays 8-bit, only MLP/expert
    weights go to 4-bit) — matches the user's "4-bit MLP + 8-bit attention"
    description.
- **Encoding format differs from `qwen3.8-27b-nvfp4`**: that checkpoint uses
  NVIDIA ModelOpt's `quantization_config` shape; this one uses the
  compressed-tensors `config_groups`/`recipe.yaml` shape (same underlying
  NVFP4 math, different metadata encoding vLLM has to parse). Both are
  documented vLLM quantization paths, but this specific encoding is
  untested on this box — if quant-kernel selection in the startup log looks
  wrong, this is the first thing to check (see `model.env`'s
  `EXTRA_VLLM_ARGS` comment).
- **Weights:** 6 safetensor shards (5 main + 1 "extra", likely vision-tower-
  or MTP-head-related), summing to **~23.0 GiB**
  (24,688,489,848 bytes / 1024³, summed from the HF repo's file listing —
  not yet cross-checked against `model.safetensors.index.json`'s own
  `total_size` field the way `qwen3.8-27b-nvfp4`'s notes did). Slightly
  larger than `qwen3.8-27b-nvfp4`'s ~21.9 GiB, in the same ballpark.
- **Reported HF "Parameters" metadata:** 21,232.1M — lower than
  `qwen3.8-27b-fp8`'s reported 27,781.4M despite being the same base model
  lineage; likely an artifact of how the Hub's parameter-counter handles
  packed sub-byte (NVFP4) tensor storage, not a real architecture
  difference. Not independently verified.
- **License:** Apache 2.0 (per repo tags).

## Why this candidate (not one of the other ~18 search hits)

A Hub search for "uncensored" + "NVFP4" variants of Qwen3.8-27B returns a
long tail of community re-quants (GGUF/llama.cpp-only, NInfer-engine-only,
AWQ/GPTQ INT4, various "Heretic"/"AEON-ULTIMATE"/"TURBO-Fable-Cold-Fusion"
branded merges of uncertain provenance). This repo is vLLM-only (see
`llm-serving-repo` memory — no SGLang/llama.cpp/GGUF/NInfer code exists
here), so anything not in a vLLM-loadable format (safetensors +
ModelOpt/compressed-tensors quant_config) was out of scope regardless of
popularity. `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` was picked because:
it's `vllm`-tagged safetensors, from the SAME publisher already trusted and
running in this repo (`qwen3.8-27b-fp8`), with the highest download/like
count among the vLLM-compatible safetensors candidates (179K downloads, 262
likes at lookup time), and the same mixed FP8/NVFP4 split already validated
architecturally on this exact box via `qwen3.8-27b-nvfp4`.

## Open questions for the smoke test (mirrors qwen3.8-27b-nvfp4's own list)

1. Does the gated download actually succeed with this box's `HF_TOKEN`
   (confirm the gate-access guess above)?
2. Clean start — no device-side assert, no OOM, at the conservative
   `GPU_MEM_UTIL=0.52`?
3. GDN Triton autotuner workaround (`DOCKER_ENV=VLLM_TRITON_FORCE_FIRST_
   CONFIG=1`) fires as expected — `[triton-autotune-disabled]` in the log?
4. Which `--moe-backend`/quant kernel does `auto` pick for the
   compressed-tensors NVFP4 encoding (untested, unlike the ModelOpt encoding
   already characterized for `qwen3.8-27b-nvfp4`)?
5. Real KV pool size / max concurrency at `GPU_MEM_UTIL=0.52` — refine from
   the actual startup log rather than assuming it matches `qwen3.8-27b-nvfp4`.
6. Tool-calling (`qwen3_coder` parser) still correct against this
   checkpoint's chat template?
7. Tokenizer truncation check (send a very long prompt, compare
   `usage.prompt_tokens` against the real token count) — same pitfall class
   caught before on other checkpoints in this repo.
8. Rough tok/s vs. both `qwen3.8-27b-nvfp4` (stock NVFP4) and
   `qwen3.8-27b-fp8` (uncensored FP8) — is the uncensored fine-tune's NVFP4
   quant as fast as the stock one, or does the fine-tuning/different
   encoding cost anything?
9. Actual refusal behavior — confirm it's as permissive as `qwen3.8-27b-fp8`
   (that's the whole point of picking this over `qwen3.8-27b-nvfp4`).

## Sources

- Model card: <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4>
- Base/sibling checkpoint already in this repo:
  <https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8>
- Stock NVFP4 sibling already in this repo, same architecture:
  `models/qwen3.8-27b-nvfp4/notes.md`,
  <https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4>
