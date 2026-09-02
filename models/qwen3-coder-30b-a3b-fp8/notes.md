# Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 — notes

**Status (2026-09-02): benchmark-tuned.** Smoke test + `overnight` sweep
(35 runs, `bench/results/overnight_20260902_153547/`).

Smoke test: serves, clean code output, **tool calling `tool_choice: "auto"`
works 3/3** (clean `tool_calls` with the `qwen3_coder` parser — Qwen2.5-Coder-32B
failed 3/3), a 105,804-token prompt answered correctly. Startup: weights
29.1 GiB / 189 s, `auto` picks **TRITON** Fp8 MoE + **FLASHINFER** attention.

## Sweep results (2026-09-02)

**~56 tok/s single-stream** flat across every config — **8.5x** the
Qwen2.5-Coder-32B dense this replaces (~6.7 tok/s, memory-bandwidth wall).
Peak observed 302 tok/s (`seqs_32` / 32 concurrent).

| axis | result | model.env |
|---|---|---|
| MoE backend | `auto` (TRITON) == forced `triton` (56/152 vs 56/155 single/batch8) | `EXTRA_VLLM_ARGS` empty |
| KV dtype | `fp8` > `auto` on batched work (batch8 155 vs 140, codegen 135 vs 124; single a wash) | `KV_CACHE_DTYPE=fp8` |
| `max_num_seqs` | no regression 1→32. Flat single/batch8 from 8 up. Only a 32-way burst rewards more (seqs_32 302 tok/s / TTFT 1.7 s vs seqs_16 214 / 76 s). Few-user on-demand → 16. | `MAX_NUM_SEQS=16` |
| prefix caching | agent-shaped: TTFT −87/−88 %, throughput +11/+46 % | `PREFIX_CACHING=1` |
| context | 8k→117, 32k→69, 65k→45, **131k→37**, 262k→19 tok/s; TTFT 2 / 5.5 / 13 / 33 / 80 s. Full attention → deep context costly. | `MAX_MODEL_LEN=131072` (262144 via one-off env) |
| `--max-num-batched-tokens` | 4096 / 8192 / default all ~132–135 tok/s on codegen — no effect | not set |
| GPU_MEM_UTIL | 0.50 @ 131072: 29.3 GiB KV / 640,672 tokens / 4.89x conc, mem peak ~72 GB | `GPU_MEM_UTIL=0.50` |

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
  `qwen3coder_tool_parser.py`; the checkpoint's embedded chat template
  (tokenizer_class `Qwen2Tokenizer`) is used directly, no `--chat-template`
  override. `tool_choice: "auto"` verified 3/3.
- **Tool calling needs xgrammar ≥ 0.2.4** — same NGC bug/fix as qwen3.8;
  `model.env` reuses the `vllm-qwen:26.07-xgrammar024` image.
- **`--moe-backend`:** `auto` picks TRITON (not cutlass). Forcing `triton`
  is identical within noise. `EXTRA_VLLM_ARGS` stays empty.

## Resolved by the sweep

Tool calling (`auto` 3/3), decode speed (~56 tok/s, 8.5x the 32B dense),
MoE backend (auto=TRITON), KV dtype (fp8), `max_num_seqs` (16, no
regression), prefix caching (big win), context default (131072 — 262144
is 2x slower + huge TTFT), `--max-num-batched-tokens` (no effect),
`GPU_MEM_UTIL` (0.50). Full numbers in the table above.

## Sources

- Model card: <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8>
- <https://qwen.readthedocs.io/en/latest/deployment/vllm.html>
- This box's image queried directly: `docker run --gpus all --entrypoint vllm
  vllm-qwen:26.07-xgrammar024 serve --help=all`
