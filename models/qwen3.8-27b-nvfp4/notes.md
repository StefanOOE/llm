# nvidia/Qwen3.8-27B-NVFP4 — notes

**Status (2026-09-14): smoke-tested, promoted to DEFAULT `gb10-reasoning`
member.** Swapped in for `qwen3.8-27b-fp8` after the smoke test (below)
showed a clean start and rough throughput clearly above the FP8 baseline.
"Default" has no formal repo mechanism — it just means this is the instance
actually started for `gb10-reasoning` traffic; `qwen3.8-27b-fp8` remains
available on-demand as the uncensored/stock-refusal alternate, and
`gpt-oss-120b` as the third member. No full `bench/matrix.yaml` sweep has
been run yet — this promotion is based on the smoke test's results, not a
formal benchmark.

## Smoke test results (2026-09-14)

Ran the full smoke-test procedure from the plan; box restored to its
starting state afterward (`qwen3.8-27b-fp8` running again).

- **Clean start.** Weight load: 20.71 GiB / 125.6s. No device-side assert,
  no OOM, at the conservative `GPU_MEM_UTIL=0.52` starting point.
- **GDN Triton autotuner workaround confirmed active and necessary-looking**:
  `[triton-autotune-disabled]` fired for all the GDN kernel variants
  (`chunk_local_cumsum_scalar_kernel`, `chunk_scaled_dot_kkt_fwd_kernel`,
  `merge_16x16_to_64x64_inverse_kernel`, `recompute_w_u_fwd_kernel`,
  `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`,
  ...), same as the FP8 checkpoint. `linear_attn` is still FP8 in this
  checkpoint, as expected — the workaround transfers directly.
- **Backends actually selected** (`auto`, not forced):
  - Attention (FP8 linear layers): `FlashInferFP8ScaledMMLinearKernel`
    (`ModelOptFp8LinearMethod`).
  - MLP/expert GEMM (NVFP4): `FlashInferCutlassNvFp4LinearKernel` — **not**
    `flashinfer_b12x`. `auto` picked the Cutlass NVFP4 kernel instead;
    `flashinfer_b12x` was never selected or attempted. Untested whether
    forcing `--moe-backend flashinfer_b12x --linear-backend flashinfer_b12x`
    would do better — `auto`'s choice already gave good results (below), so
    this wasn't pursued further this round.
  - GDN prefill: Triton/FLA kernel (`requested=auto`), same as FP8 checkpoint.
  - General attention: FLASHINFER.
- **KV cache**: 37.6 GiB available, 978,310 tokens of pool at
  `GPU_MEM_UTIL=0.52` — comfortable headroom, room to raise later if pursued.
- **Functional checks — all pass:**
  - `/v1/models`: correct identity (`gb10-reasoning`, root
    `nvidia/Qwen3.8-27B-NVFP4`).
  - Plain chat: correct, coherent.
  - Tool calling (`qwen3_coder` parser): correct `tool_calls` structure and
    arguments on the first try — chat template compatibility confirmed.
  - Vision/OCR (reused `qwen3.8-27b-fp8/bench/ocr_test.png`): read the test
    invoice image correctly.
  - **Tokenizer truncation pitfall: refuted for this checkpoint.** Sent a
    prompt tokenizing to 66,911 tokens; `usage.prompt_tokens` reported
    exactly 66,911 — no silent truncation at 2048 or anywhere else. The
    community report the user surfaced does not apply here (consistent with
    the pre-download analysis: that report is about a different, unrelated
    model).
- **Rough throughput vs. FP8 baseline** (single-stream, MTP=3, 300 output
  tokens each, casual timing — not a formal bench run):
  - Run 1: 300 tokens / 16.59s ≈ **18.1 tok/s**
  - Run 2: 300 tokens / 11.41s ≈ **26.3 tok/s**
  - Both clearly above the FP8 baseline's tuned single-stream figure
    (~14 tok/s w/ MTP=3). Promising, though only 2 samples — noisy, no
    concurrency/batch probing done, no accept-rate instrumentation checked.
    A real `bench/matrix.yaml` sweep would be needed before trusting this as
    a stable number.
- **Not tested this round:** `--kv-cache-dtype nvfp4`, `flashinfer_b12x`
  forced, batch/concurrent throughput, MTP accept-rate under NVFP4 weights,
  quality/behavior eval (stock vs. uncensored — this checkpoint **will**
  refuse things the FP8 checkpoint currently complies with; not exercised in
  this smoke test since none of the test prompts touched that territory).

**Recommendation:** results are good enough to justify a real
`bench/matrix.yaml` sweep if there's interest in adopting this as (or
alongside) the `gb10-reasoning` default — but that's a separate decision
from this smoke test, especially given the stock/refusal behavior trade-off
against the currently-uncensored FP8 checkpoint.

## Full bench sweep results (2026-09-14, `gap_close_nvfp4`)

Formal sweep, `models/qwen3.8-27b-nvfp4/bench/matrix.yaml`, suite
`gap_close_nvfp4` (9 configs, 31 (config,workload) runs). Ran with
`VLLM_TRITON_FORCE_FIRST_CONFIG=1` in the sweep's own `env:` block — the
GDN workaround is NOT auto-inherited from `model.env`'s `DOCKER_ENV` by
`run.py`, had to be added explicitly (see the matrix.yaml header). Full
data: `bench/results/gap_close_nvfp4_20260914_193918/summary.csv` +
`report.md`. `gb10-reasoning` was offline for the ~3h duration (bench
harness reuses the serving port); `qwen3.8-27b-nvfp4` was restarted as the
live default immediately after.

**MTP token sweep** (`mtp_sweep_nvfp4`, at the deployed production combo:
ctx131072/gmu0.52/seqs32/kv=fp8), `output_tok_s` / `spec_accept_len`:

| config (MTP) | single_stream | batch8_balanced | codegen_decode | saturate32 |
|---|---|---|---|---|
| off | 12.46 / — | 84.52 / — | 84.8 / — | 217.44 / — |
| 1 | 18.38 / 1.76 | 109.21 / 1.74 | 122.27 / 1.84 | 257.56 / 1.74 |
| 2 | 19.88 / 2.06 | 122.87 / 2.23 | 139.65 / 2.51 | 275.57 / 2.33 |
| **3 (deployed)** | **21.41 / 2.38** | 118.4 / 2.65 | **146.4 / 2.90** | 263.4 / 2.62 |
| 4 | 21.39 / 2.51 | 111.34 / 3.05 | 120.12 / 3.20 | **`bench_errored`** — reproduced twice (fresh cold container both times), 32/64 requests returned zero output tokens in ~6s. Root cause not fully diagnosed (`run.py` doesn't capture server-side logs *during* a run, only at startup — no crash/assert visible in the captured log); NOT a transient fluke. |

**Conclusion: MTP=3, the already-deployed default, is confirmed correct** —
best or tied-best on 3/4 workloads, and MTP=4 shows diminishing returns on
the workloads it *did* complete (worse than MTP=3 on batch8_balanced and
codegen_decode despite higher accept_len) plus the reproducible saturate32
failure. MTP=4 should not be used with this checkpoint at
`max_num_seqs=32`/high concurrency until that failure is root-caused.
`MTP_TOKENS=3` in `model.env` stays as-is, now backed by real data instead
of just inherited from the FP8 sweep.

**Head-to-head vs. FP8 at FP8's exact benched combo** (ctx65536/gmu0.65/
seqs32/mtp3/kv=fp8 — reused from `qwen3.8-27b-fp8/bench/results/
prod_mtp_sweep_65k_20260831_215815/summary.csv`'s `prod_mtp3` row, not
re-run here):

| workload | FP8 (prod_mtp3) | NVFP4 (nvfp4_65k_kv_fp8) | delta |
|---|---|---|---|
| single_stream | 14.43 | 23.66 | +64% |
| batch8_balanced | 86.51 | 128.22 | +48% |
| codegen_decode | 87.29 | 153.18 | +76% |
| saturate32 | 210.21 | 259.05 | +23% |

NVFP4 is faster than FP8 at every matched workload, not just the earlier
casual samples — margin is largest on latency-bound/low-concurrency
workloads (single_stream, codegen_decode) and narrows at saturation
(saturate32), consistent with the 4-bit MLP weights mainly helping
memory-bandwidth-bound decode rather than compute-bound batched prefill.

**KV cache dtype A/B** (ctx65536/gmu~0.65-0.70/seqs32/mtp3):

| kv_cache_dtype | single_stream | ctx64k_load (28k prefix, deep-context stress) |
|---|---|---|
| fp8 (deployed default) | 23.66 | 40.73 |
| auto (bf16) | 23.13 | 37.6 |
| **nvfp4** | **failed to start** — `ValueError: --kv-cache-dtype nvfp4 requires sm100f, please try a different dtype or remove`. **Hardware-gated, not a config issue**: this vLLM build's `nvfp4` KV-cache dtype needs SM100 (datacenter Blackwell); DGX Spark/GB10 is SM121. Same class of rejection as gpt-oss's MXFP4 backend and the FP8 MoE CUTLASS backend (vLLM #43507) — a recurring SM12x-vs-SM100 gap on this box. **Not usable on this hardware at all**, not worth revisiting. |

fp8 vs. auto(bf16): fp8 KV is marginally faster at both single-stream and
deep context (~2-8%), consistent with `qwen3.8-27b-fp8`'s own
`kv_dequant_cliff_check` finding that fp8 KV doesn't hit a real dequant
penalty on this bandwidth-rich unified-memory box. `KV_CACHE_DTYPE=fp8`
in `model.env` stays as-is.

**MoE/linear-backend probe**: `nvfp4_backend_b12x` (forcing
`--moe-backend flashinfer_b12x --linear-backend flashinfer_b12x`) **failed
to start**: `ValueError: --linear-backend=flashinfer_b12x was requested but
no 'flashinfer_b12x' kernel exists for this layer type.` Root cause:
`--linear-backend` applies backend-wide, including to this checkpoint's
FP8 `linear_attn` layers — but `flashinfer_b12x` only has NVFP4 kernels, so
it errors on the FP8 attention linear layers. **Not a hardware gap** like
the KV-dtype failure above — plausible that `--moe-backend flashinfer_b12x`
*alone* (without forcing `--linear-backend`) would work and only affect the
NVFP4 MoE/MLP weights, but that combination wasn't tried this round.
`auto`'s pick (`FlashInferCutlassNvFp4LinearKernel`, confirmed at smoke-test
time) already beats the FP8 baseline substantially (table above), so this
wasn't pursued further. `EXTRA_VLLM_ARGS` in `model.env` stays empty.

**Net result**: the 2026-09-14 promotion to DEFAULT `gb10-reasoning` member
is now backed by a full sweep, not just the smoke test's casual timing.
`model.env`'s inherited defaults (`MTP_TOKENS=3`, `KV_CACHE_DTYPE=fp8`,
empty `EXTRA_VLLM_ARGS`) all check out as the right choices for this
checkpoint too — nothing to change. Two loose ends: the reproducible
MTP=4/saturate32 failure (not root-caused, but MTP=4 isn't the default so
low priority) and whether `--moe-backend flashinfer_b12x` alone (without
`--linear-backend`) beats `auto` (untested, `auto` already wins comfortably
so also low priority).

## 2026-09-15: MAX_MODEL_LEN raised to native ceiling (262144)

Deliberate choice, made after explaining the trade-off: `--gpu-memory-
utilization` carves a FIXED slice of the pool regardless of
`max_model_len` (repo-wide quirk, see README), so raising context doesn't
grow the KV pool — it only raises the ceiling one request can claim from
it, which (a) lowers max concurrency for the same pool and (b) raises the
cold-start profiling peak (the class of failure behind the 2026-09-10
OOM-crash-loop). Verified after restart, at the unchanged `GPU_MEM_UTIL=
0.52`:

- KV pool: 37.57 GiB / 1,060,291 tokens — essentially the same pool size as
  at 131072 (was 37.6 GiB / 978,310 tokens at smoke-test time), confirming
  the pool itself doesn't shrink or grow with `max_model_len`.
- **Maximum concurrency: 4.04x** (was 9.94x in the `gap_close_nvfp4` sweep
  at 131072, same `GPU_MEM_UTIL`) — concurrency dropped by more than half,
  exactly the predicted trade-off, not re-swept at this setting.
- Clean start, no device-side assert, no OOM. Functional check (plain chat)
  passed.

Previous value (131072) kept commented out in `model.env` next to the new
one as a quick rollback if 4.04x concurrency proves too tight in practice.

Role on this box: evaluation of the *official* NVFP4 quant of Qwen3.8-27B as
a possible alternative to the FP8 checkpoint currently serving
`gb10-reasoning`. Exploratory — not committed to adoption. `model.env`'s
tuned-looking knobs are mostly **inherited starting points** copied from
`qwen3.8-27b-fp8`'s benchmarked values (same base architecture), not yet
re-verified for this checkpoint.

## What it is

- **Publisher:** NVIDIA, Apache 2.0. `base_model: Qwen/Qwen3.8-27B` (the
  **stock**, non-abliterated release — not the `orcarouter` uncensored
  derivative this box's FP8 checkpoint uses). Produced with
  `nvidia-modelopt v0.48.0`.
- **Architecture:** `Qwen3_5ForConditionalGeneration` — identical shape to
  `qwen3.8-27b-fp8`: 64 layers, hybrid attention (16 full-attention + 48
  Gated-DeltaNet linear, interval 4), native vision + video tower, MTP head
  (`mtp_num_hidden_layers: 1`). Only the weight precision differs.
- **Quant scheme — mixed NVFP4/FP8, verified from the checkpoint's own
  `hf_quant_config.json`** (ModelOpt, `quant_algo: MIXED_PRECISION`):
  - `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` (the GDN attention
    layers) → **FP8**
  - `mlp.{gate,up,down}_proj` (per-layer MLP / expert weights) → **NVFP4**,
    `group_size: 16`
  - Matches the user's "4-bit MLP + 8-bit attention" description exactly.
- **Weights:** ~21.9 GiB (`model.safetensors.index.json`: `total_size`
  21,921,428,072 bytes; 3 shards) vs `qwen3.8-27b-fp8`'s ~28.95 GiB — **~24%
  smaller**. Real but modest: only the MLP/expert weights actually go to
  4-bit; attention, norms, embeddings, lm_head, vision tower stay at FP8/BF16.
- **Context:** same native ceiling as the FP8 checkpoint (262144, no YaRN
  needed for 131072).
- **License:** Apache 2.0 (per NVIDIA's model card — "GOVERNING DOWNLOAD
  TERMS: Apache 2.0", "ready for commercial or non-commercial use").

## Pitfalls checked against this specific checkpoint (2026-09-14)

The user surfaced 3 external reports of NVFP4-on-DGX-Spark/vLLM issues.
Checked against this repo, this box's vLLM image, and the checkpoint's own
metadata before writing this scaffold:

1. **`ngram_embedding`/`weight_scale` KeyError ("showstopper" load crash) —
   does NOT apply.** That failure is reported against
   `Qwen3.8-Flash-Next-NVFP4` (~180B MoE, has a large n-gram/PLE embedding
   table) — a different, much larger model already excluded from this box
   (see the `llm-serving-repo` memory note: ~186 GB in any quant, needs
   TP ≥ 8). Verified directly: `nvidia/Qwen3.8-27B-NVFP4`'s own
   `model.safetensors.index.json` weight map has **no** `ngram`/PLE-table
   keys — only the standard per-layer `linear_attn.*` / `mlp.*` weights.
2. **"Needs a newer NGC image / cu130-nightly" — already satisfied.** This
   box runs `nvcr.io/nvidia/vllm:26.07-py3`, past the cited 26.02 floor. vLLM
   in this image (checked directly in the container) ships real, non-stub
   NVFP4/ModelOpt code: `model_executor/layers/quantization/modelopt.py`,
   `.../compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`,
   `.../compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py`, plus
   an `nvfp4_emulation_moe.py` / `nvfp4_emulation_utils.py` fallback path
   (for hardware without native FP4 tensor-core support, if that's ever
   relevant here).
3. **Tokenizer silently truncating prompts at 2048 tokens — genuinely
   unverified, check at smoke test.** Can't confirm or deny from metadata
   alone; the long-context smoke-test step compares the response's
   `usage.prompt_tokens` against the actual sent token count (same technique
   that caught the `deepseek-coder-v2-lite-fp8` tokenizer bug).

Not independently checkable without a real start: "illegal instruction"
crashes from kernels compiled only for x86 hosts or datacenter-Blackwell
(SM100), a plausible risk class given this box's history of SM121-specific
kernel gaps (the GDN Triton bug below; gpt-oss's MXFP4 backend rejections)
but not something metadata inspection can confirm — that's what the smoke
test's log-reading step is for.

## Applied proactively (not rediscovered)

**GDN Triton autotuner OOM workaround.** `qwen3.8-27b-fp8` hit a device-side
assert / OOM on 2026-09-10 in the exact same Gated-DeltaNet linear-attention
Triton kernel path (vLLM #36598/#33857 — GB10/SM121 isn't warmed during the
V1 profiling run, so the first real GDN prefill fires the Triton autotuner
with no memory left). This checkpoint's `linear_attn.*` layers stay FP8 (the
NVFP4 conversion only touches MLP weights), so the same Triton kernel code
path runs unchanged — the bug almost certainly still applies. `model.env`
sets `DOCKER_ENV=VLLM_TRITON_FORCE_FIRST_CONFIG=1` from the first start
rather than waiting to crash into it. Confirm active at smoke test: the
engine log should print `[triton-autotune-disabled]` for the GDN kernels,
with no device-side assert.

## Open questions for the smoke test

1. Does it start clean at all (weight loading, quant-kernel selection, GDN
   workaround)?
2. Which `--moe-backend` / `--linear-backend` does `auto` actually pick?
   This vLLM build has a `flashinfer_b12x` backend explicitly documented as
   *"FlashInfer CuteDSL fused MoE for SM12x (RTX Pro 6000 / DGX Spark)"* for
   **NVFP4** specifically — purpose-built for this exact chip + quant format
   (contrast: it explicitly *rejects* gpt-oss's MXFP4 — a quant-type
   mismatch there, not a hardware gap). If `auto` doesn't pick it, try
   forcing: `EXTRA_VLLM_ARGS="--moe-backend flashinfer_b12x --linear-backend flashinfer_b12x"`.
3. Real KV pool size / max concurrency at `GPU_MEM_UTIL=0.52` (conservative
   starting guess — refine from the actual startup log, don't repeat the
   2026-09-10 OOM-crash-loop mistake on an unbenchmarked checkpoint).
4. Does `--kv-cache-dtype nvfp4` (a real option in this vLLM build, unique
   relevance to this checkpoint) work, and is it worth it over `fp8`?
5. Tool-calling: does `qwen3_coder` still parse this checkpoint's chat
   template correctly? (inherited default, not yet verified for this repo).
6. MTP accept-rate under NVFP4 MLP weights — inherited `MTP_TOKENS=3` from
   the FP8 sweep, not re-verified.
7. Rough tok/s vs. the FP8 baseline (single-stream ~14 tok/s w/ MTP=3) — is
   NVFP4 faster, comparable, or slower on this MoE+hybrid-attention model?
   (Unlike the Qwen2.5-Coder-32B *dense* bandwidth-wall case, this model is
   already MoE, so the win from 4-bit vs 8-bit MLP weights is less certain
   — real numbers only, no armchair prediction.)
8. Quality/behavior: stock (non-abliterated) — refusals expected where the
   uncensored FP8 checkpoint would comply. Real trade-off if this is ever
   considered for the identity's default slot, separate from performance.

## Sources

- Model card: <https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4>
- Base model: <https://huggingface.co/Qwen/Qwen3.8-27B>
- NVIDIA Model Optimizer: <https://github.com/NVIDIA/Model-Optimizer>
- This box's image queried directly: `docker run --gpus all --entrypoint vllm
  vllm-qwen:26.07-xgrammar024 serve --help=all`; vLLM source paths under
  `vllm/model_executor/layers/quantization/` inspected directly in the
  container.
- GDN Triton workaround: `models/qwen3.8-27b-fp8/model.env` /
  `models/qwen3.8-27b-fp8/notes.md` "2026-09-10", vLLM issues #36598 / #33857.
