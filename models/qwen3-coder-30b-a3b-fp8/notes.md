# Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 — notes

**Status (2026-09-02): scaffolded + smoke-tested, NOT yet benchmark-tuned.**
Smoke test (qwen3.8 stopped):
- serves, `/v1/models` ctx 262144, clean code output.
- **Decode ~57 tok/s** sustained (600 tok in 10.5 s) — **8.5x** the
  Qwen2.5-Coder-32B dense it replaces (~6.7 tok/s). This is the point.
- **Tool calling `tool_choice: "auto"` works 3/3** (clean `tool_calls`,
  name + args) with the `qwen3_coder` parser — Qwen2.5-Coder-32B failed 3/3.
- Long context: a 105,804-token prompt answered correctly, 66 s TTFT.
- Startup: weights 29.1 GiB / 189 s. `auto` picked **TRITON** Fp8 MoE +
  **FLASHINFER** attention.
- **KV pool thin at deep context:** at `GPU_MEM_UTIL 0.40`, 16.56 GiB /
  361,712 tokens / **1.38x concurrency @ 262144**. Bumped default to 0.50;
  the `context_scaling` sweep settles GMU vs the 262144-vs-131072 default.

`model.env` numeric knobs are still starting points — run `bench/matrix.yaml`'s
`overnight` suite and fold winners in.

Role on this box: the **quality** coding model (the **performance** one is
`deepseek-coder-v2-lite-fp8`). On-demand only — **not** a systemd service;
`qwen3.8-27b-fp8` must be stopped first (memory). See the repo README.

## Why this model, not Qwen2.5-Coder-32B

The plan originally called for `Qwen2.5-Coder-32B-Instruct-FP8`. It was
scaffolded, smoke-tested and partially swept (2026-09-01/02) — and it hit the
**GB10 memory-bandwidth wall**: **147 ms/token / ~6.7 tok/s single-stream**,
dead flat across every config (KV dtype, `max_num_seqs`, backend). A 32B
*dense* FP8 model streams ~32 GiB of weights per token; at ~217 GB/s
effective LPDDR5x that's ~147 ms, and no serving knob moves it. Batch was
fine (~50 tok/s aggregate at 8 concurrent) but interactive/agentic use at
7 tok/s is not.

`Qwen3-Coder-30B-A3B` is **MoE** (30.5B total, **3.3B active/token**, 128
experts / 8 active) — only a fraction of the weights move per token, so it
decodes several times faster, and it benchmarks *above* Qwen2.5-Coder-32B on
most coding evals. This is the same MoE trade the box already makes for
`qwen3.8-27b-fp8` and `gpt-oss-120b`.

## What it is

- **Publisher:** Qwen (Alibaba), Apache 2.0. Official FP8 release.
- **Architecture:** `Qwen3MoeForCausalLM` (`qwen3_moe`) — 48 layers, hidden
  2048, 128 experts / 8 active per token, `moe_intermediate_size` 768,
  `decoder_sparse_step` 1 (every layer is MoE). GQA: 32 query / 4 KV heads,
  head_dim 128. Full attention, no sliding window.
- **Quant:** `quant_method: fp8`, e4m3, `weight_block_size [128,128]` —
  block FP8, the same scheme as `qwen3.8-27b-fp8`. `activation_scheme:
  dynamic`. Per-layer layernorms, MoE gates, and `lm_head` stay BF16.
- **Weights:** ~31.2 GiB / 4 shards.
- **Context:** **262,144 native** (`rope_scaling: null`, `rope_theta` 1e7) —
  no YaRN needed, `model.env` ships `MAX_MODEL_LEN=262144`.
- **No MTP head** — no speculative decoding. **No vision.**
- **Not a thinking model** — no reasoning channel.

## vLLM specifics (this box: NGC `26.07-py3`, vLLM 0.24.0)

- **`VLLM_USE_DEEP_GEMM=0`** applies (box-wide sm_121 assert) — also rules
  out `--moe-backend deep_gemm`.
- **Do NOT pass `--quantization`** — read from `quantization_config`.
- **Tool calling:** dedicated parser `qwen3_coder` (in this image's
  `--tool-call-parser` enum). The repo ships `chat_template.jinja` +
  `qwen3coder_tool_parser.py`; vLLM also bundles
  `/opt/vllm/vllm-src/examples/tool_chat_template_qwen3coder.jinja`.
  **VERIFY at first start** that `tool_choice: "auto"` returns non-null
  `tool_calls` — Qwen2.5-Coder-32B failed this (emitted the wrong wrapper);
  Qwen3-Coder is the model `qwen3_coder` was built for, so it should work.
- **Tool calling needs xgrammar ≥ 0.2.4** — same NGC bug/fix as qwen3.8;
  `model.env` reuses the `vllm-qwen:26.07-xgrammar024` image.
- **`--moe-backend`:** `flashinfer_cutlass` was rejected for `qwen3.8-27b`'s
  block-FP8 MoE on SM120/121 (vLLM #43507) — expect the same here. Probe
  `auto` vs `triton`, correctness-gated.

## Open questions for the sweep

1. **RESOLVED: tool calling `tool_choice: "auto"`** — works 3/3, no override.
2. **RESOLVED: decode speed** — ~57 tok/s, clears the 32B-dense bar 8.5x.
3. **RESOLVED: MoE backend** — `auto` picks TRITON; `moe_backend_probe`
   re-checks vs forced.
4. **Context default** — keep 262144 (needs GMU ~0.5 for headroom) or drop
   to 131072? `context_scaling` decides.
5. **KV dtype** — `fp8` vs `auto` (`kv_sweep`).
6. **`max_num_seqs`** sweet spot (`seqs_sweep`) — MoE regression watch.
7. **Prefix caching** win size on coding traffic.
8. **`GPU_MEM_UTIL`** — 0.50 placeholder; derive from the sweep.

## Sources

- Model card: <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8>
- <https://qwen.readthedocs.io/en/latest/deployment/vllm.html>
- This box's image queried directly: `docker run --gpus all --entrypoint vllm
  vllm-qwen:26.07-xgrammar024 serve --help=all`
