# openai/gpt-oss-120b — notes

**Status (2026-08-31): smoke-tested and correctness-verified on this box.**
The SM121 "null Harmony token" bug (below) does **not** reproduce here —
`auto` picks the MARLIN MXFP4 backend and both `content` and `reasoning`
come back non-null and coherent across repeated requests, plain chat and
tool-calling alike. Benchmark sweep (`bench/matrix.yaml`) is next; until it
runs, `model.env`'s numeric knobs (context/seqs/KV dtype) are still
starting-point values, not benchmark results, unlike qwen3.8-27b-fp8's.

## What it is

- **Publisher:** OpenAI, Apache 2.0, released natively pre-quantized (no BF16
  release exists at this size — `openai/gpt-oss-120b` *is* the checkpoint).
- **Architecture:** `GptOssForCausalLM` (`gpt_oss`) — 36 layers, hidden 2880,
  MoE with 128 local experts / 4 active per token, GQA (64 query heads / 8 KV
  heads, head_dim 64), alternating sliding-window (window=128) / full
  attention (18 of each) — every other layer is cheap regardless of context
  length, unlike qwen3.8's mostly-full-attention stack.
- **Context:** trained at 4096, extended to **131,072** via YaRN
  (`rope_scaling: {type: yarn, factor: 32}`) — this is the vendor's
  "128k context" claim, not a from-scratch long-context model.
- **Weights:** native MXFP4 (`quantization_config.quant_method: mxfp4`),
  `modules_to_not_convert`: attention, router, embeddings, lm_head (those stay
  BF16). ~63 GiB total / 14 safetensors shards. A BF16 dequant would be
  ~235 GiB — does not fit the 128 GB unified pool, so MXFP4 is not optional
  here (unlike qwen's FP8, which was a bandwidth-speed choice over a BF16
  checkpoint that *did* fit).
- **No speculative-decoding head** — MTP/EAGLE do not apply; `MTP_TOKENS`
  stays 0 in `model.env`, kept only because `lib.sh` reads it unconditionally.
- **No vision tower** — text-only, `LANGUAGE_MODEL_ONLY=1` in `model.env`.
- **Parsers** (Harmony response format): `--reasoning-parser openai_gptoss`,
  `--tool-call-parser openai`. Confirmed present in this box's image
  (`vllm.reasoning.gptoss_reasoning_parser`, registered as `openai_gptoss`;
  `openai` is a listed `--tool-call-parser` choice) — see "vLLM specifics".

## vLLM specifics (this box: NGC `26.07-py3`, vLLM 0.24.0)

- **`VLLM_USE_DEEP_GEMM=0` still applies** (box.env, box-wide) — unrelated to
  gpt-oss specifically but still mandatory on sm_121.
- **Do NOT pass `--quantization mxfp4`** — read from the checkpoint's
  `config.json`, same rule as qwen's FP8 (`model.env` comment there).
- **`--moe-backend`** in this vLLM build has a backend literally named for
  this hardware: `flashinfer_b12x` — "Use FlashInfer CuteDSL fused MoE for
  SM12x (RTX Pro 6000 / DGX Spark)" (confirmed via `vllm serve --help=all` in
  the box's own image, 2026-08-29). `auto` may now select it correctly.
- **`--moe-backend marlin`** also exists in this image's `--help=all` enum.

### The SM121 "null Harmony token" bug — VERIFIED NOT REPRODUCING (2026-08-31)

Multiple independent DGX-Spark writeups (community forum threads, blog posts,
one filed vLLM issue) describe the *same* failure on SM121/GB10 across vLLM
0.16.0rc2 through 0.20.1 (NGC `26.03`-`26.05`): the MoE backend that gets
auto-selected on SM121 falls back to an SM80-only Marlin MXFP4 kernel, which
produces a wrong logit for the very first assistant token. Since gpt-oss's
Harmony format expects that first token to be a control token
(`<|channel|>`, id 200005), the wrong token silently breaks the parser and
every response comes back `content: null, reasoning: null` — despite the
server reporting completion_tokens > 0. No accepted fix was visible in the
vLLM issue tracker as of the writeups above; the workaround everywhere was to
force a different backend explicitly (`VLLM_MXFP4_BACKEND=marlin` env var on
older versions using the pre-`--moe-backend`-flag CLI, or the `--moe-backend`
flag directly on newer ones) and, in some threads, `--attention-backend
TRITON_ATTN` (FlashInfer attention was separately reported broken/slow for
gpt-oss's attention-sink layers on some of those versions).

**This box's image (26.07-py3 / vLLM 0.24.0) is newer than every version in
those reports.** Verification (2026-08-31, real `./serve start` +
manual curl, not the automated bench harness — see matrix.yaml's header on
why that harness alone can't catch this):

- `auto` MoE backend selection actually picked **MARLIN** (log line:
  `Using 'MARLIN' Mxfp4 MoE backend`) — the exact backend named in the bug
  reports, not the newer `flashinfer_b12x` one. Attention backend: `TRITON_ATTN`.
- Plain chat and a `tools`-bearing request both returned non-null, coherent
  `content`/`reasoning` — repeated 3x on a second prompt for stability, same
  result every time (`Paris` / `Paris.` / `Paris`, sensible reasoning trace
  each time).
- **Conclusion: the bug does not reproduce on this image**, even on the
  backend it was originally reported against. Whatever fixed it landed
  somewhere in the vLLM version range between the writeups (0.16.0rc2-0.20.1)
  and 0.24.0 — no need to force `flashinfer_b12x` or fall back to
  `TRITON_ATTN` explicitly; `EXTRA_VLLM_ARGS` stays empty in `model.env`.

One unrelated warning seen at startup, apparently benign (reasoning/content
came back fine regardless): `Auto-initialization of reasoning token IDs
failed. Please check whether your reasoning parser has implemented the
reasoning_start_str and reasoning_end_str.`

The escalation order below is now dead code for THIS image (kept in
`model.env` only as a documented fallback in case a future image/driver
update regresses this):
1. ~~`--moe-backend flashinfer_b12x`~~ -- **invalid**, see next section
2. `--moe-backend marlin --attention-backend TRITON_ATTN`

### `flashinfer_b12x` is not a valid MXFP4 backend (found 2026-08-31)

`vllm serve --help=all`'s `--moe-backend` enum lists `flashinfer_b12x` as
"Use FlashInfer CuteDSL fused MoE for SM12x (RTX Pro 6000 / DGX Spark)" --
looked like the obvious choice for this exact hardware. It isn't: passing
it crashes engine-core init immediately (`bench/matrix.yaml`'s
`backend_sweep`, `backend_flashinfer_b12x` config, before the fix below):

```
ValueError: moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE.
Expected one of ['deep_gemm', 'flashinfer_trtllm', 'flashinfer_trtllm_afp8',
'flashinfer_cutlass', 'flashinfer_cutlass_afp8', 'triton', 'triton_unfused',
'humming', 'marlin', 'aiter', 'aiter_mxfp4_fp8', 'aiter_mxfp4_mxfp4', 'xpu',
'cpu', 'emulation'].
```

So `flashinfer_b12x` is real, but scoped to some other fused-MoE quant path
(not MXFP4) -- `--help=all`'s per-hardware label doesn't imply per-quant
compatibility. `matrix.yaml` now benchmarks `flashinfer_cutlass` in its
place (accepted for MXFP4, plausibly the actual "modern SM12x" alternative
to Marlin); `deep_gemm` is left untested since `VLLM_USE_DEEP_GEMM=0` is
mandatory box-wide (separate sm_121 assert, see `common/box.env`) and
forcing it as `--moe-backend` would fight that setting. `aiter*` are
AMD/ROCm kernels, not applicable here.

### Partial-download vs. vLLM's snapshot completeness check (2026-08-31)

First start attempt failed: `IncompleteSnapshotError: ... 11 file(s) are
missing (metal/model.bin, original/config.json, ...)`. Cause: our download
deliberately excludes `metal/` and `original/` (Apple/OpenAI-native formats
vLLM doesn't read), but vLLM 0.24.0's `get_model_path()` still calls HF's
`snapshot_download()` even under `HF_HUB_OFFLINE=1`, which validates the
local cache against the *full* upstream file listing and raises on any
"missing" file — including ones excluded on purpose.

Fix: that function's actual first line is `if os.path.exists(model): return
model` — a local path skips the whole hub/completeness check. `model.env`
now resolves the cached snapshot dir from `HF_CACHE/hub/.../refs/main` and
passes it via `lib.sh`'s new `VLLM_MODEL_PATH` (overrides what's handed to
`vllm serve`; `MODEL` itself stays the repo id everywhere else — preflight,
banner, `hf download` hints). See `lib.sh`'s header comment for the general
mechanism.

### Reported throughput elsewhere (NOT this box — different vLLM versions/images)

For context only, not a target: community numbers on GB10 single-box range
from ~37-39 tok/s (stock NGC 26.0x path, single-stream) up to ~59-80 tok/s
with hand-patched/forked builds (`--load-format fastsafetensors`,
non-stock kernels, or two-box tensor-parallel setups). None of these are
directly comparable to a stock NGC 26.07 run — treat our own bench numbers as
authoritative once they exist, not these.

## Sources

- Model card / config: <https://huggingface.co/openai/gpt-oss-120b>
- vLLM recipe (generic, not SM121-specific): <https://recipes.vllm.ai/openai/gpt-oss-120b>
- Root-cause bug report: <https://github.com/vllm-project/vllm/issues/37030>
- NVIDIA DGX Spark dev forum: "vLLM on GB10: gpt-oss-120b MXFP4 slower than
  SGLang/llama.cpp", "vLLM 0.17.0 MXFP4 Patches for DGX Spark", "Deterministic
  gpt-oss-120b using vLLM on a DGX Spark", "Solved - running gpt oss 120b with
  two sparks" (forums.developer.nvidia.com)
- Practitioner writeups: ai-muninn.com "Running a 120B Model on DGX Spark at
  60 tok/s — Zero API Cost, Six Bugs"; conselara.dev "Running gpt-oss-120b on
  a Single DGX Spark"
- This box's own image, queried directly: `docker run --gpus all --entrypoint
  vllm nvcr.io/nvidia/vllm:26.07-py3 serve --help=all`, and reasoning-parser
  registry inspected via `python3 -c "from vllm.reasoning import ..."`.
