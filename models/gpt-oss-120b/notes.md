# openai/gpt-oss-120b — notes

**Status (2026-08-29): weights downloading, not yet smoke-tested on this box.**
Everything below is either a checkpoint fact (config.json) or a claim from
third-party writeups about *other* vLLM versions/images on the same GB10
hardware — treat the latter as hypotheses to verify, not settled config.
Once a bench pass runs, replace this header and `model.env`'s "PENDING
BENCHMARK" values with our own numbers.

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

### The SM121 "null Harmony token" bug — unverified on this image

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
those reports**, and — notably — already ships the `flashinfer_b12x` backend
built for exactly this GPU family, which didn't exist in the versions where
the bug was reported. It is plausible the bug is already fixed here via
`auto`-selection, but this is unverified — nobody's writeup covers 0.24.0 /
26.07 specifically. **First smoke test after weights land must check for
this exact symptom** (send a real chat request, confirm `content` is
non-null and coherent — not just that the server returns HTTP 200) before
trusting any config, benchmarked or not.

If it reproduces, escalate through `model.env`'s `EXTRA_VLLM_ARGS` in this
order (documented there too):
1. `--moe-backend flashinfer_b12x` (explicit, in case `auto` guesses wrong)
2. `--moe-backend marlin --attention-backend TRITON_ATTN` (matches the
   community workaround combination most consistently reported to work)

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
