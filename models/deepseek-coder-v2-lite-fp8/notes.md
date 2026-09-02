# RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8 — notes

**Status (2026-09-02): benchmark-tuned.** Serves cleanly (clean detokenization
after the tokenizer fix below). The `overnight` sweep (31 runs,
`bench/results/overnight_20260902_050212/`) confirmed every `model.env`
default — no value changed, because the starting points were right (or
forced by what works on this box). Three start-blocking issues were found +
fixed during the smoke test (trust-remote-code, `fp8_ds_mla`, tokenizer).

## Sweep results (2026-09-02)

Single-stream **~69 tok/s** (fast — the "performance" model earns the name;
qwen3.8-27b ≈ 28, gpt-oss-120b ≈ 12). tpot ~14.4 ms. All at `GPU_MEM_UTIL
0.30` (peak unified-mem use ~44 GiB, ~52 GiB at ctx 131k).

| axis | result | model.env |
|---|---|---|
| MoE backend | `auto` == `triton` (69 / 172 tok/s single/batch8, identical). `cutlass` disabled for this config, `deep_gemm` box-blocked. | `EXTRA_VLLM_ARGS` = tokenizer only (auto) |
| KV dtype | `auto` is the **only** working option (see below) | `KV_CACHE_DTYPE=auto` |
| `max_num_seqs` | sweet spot **8** — codegen_decode 68→97→128→**166**→167 tok/s at 1/2/4/8/16, no regression past 8 (unlike gpt-oss) | `MAX_NUM_SEQS=8` |
| prefix caching | big win — `agent_turn_solo` TTFT 1108→126 ms (−89%), `agent_turn_parallel` 2070→277 ms (−87%), throughput +11/+34% | `PREFIX_CACHING=1` |
| context scaling | 8k→154, 32k→99, 65k→74, 131k→47 tok/s; TTFT 1.5/5.9/10.7/20.7 s. MLA keeps 131k usable. | `MAX_MODEL_LEN=131072` |
| GPU_MEM_UTIL | 0.30 peaks ~44–52 GiB → does **not** fit alongside a live qwen3.8 (0.65 ≈ 83 GiB) + bge (15 GiB) = ~148 GiB > 128. On-demand only. | `GPU_MEM_UTIL=0.30` |

Role on this box: the **performance** coding model (the **quality** one is
`qwen3-coder-30b-a3b-fp8`). On-demand only — **not** a systemd service, and
it does **not** co-reside with a live `qwen3.8-27b-fp8` (0.30 peaks
~44–52 GiB unified; qwen3.8 + bge already use ~98). Stop qwen3.8 first, same
as the other on-demand models.

## What it is

- **Base:** `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` (DeepSeek license),
  FP8-quantized by RedHat/Neural Magic.
- **Architecture:** `DeepseekV2ForCausalLM`.
  - **MoE:** 64 routed + 2 shared experts, **6 active per token**, 27 layers,
    `first_k_dense_replace: 1` (layer 0 dense), `moe_intermediate_size` 1408.
    15.7B total / ~2.4B active params.
  - **MLA attention:** `kv_lora_rank` 512, `qk_nope_head_dim` 128,
    `qk_rope_head_dim` 64, `v_head_dim` 128, 16 heads. KV cache is a small
    per-token latent, **not** full per-head K/V — long context is cheap.
- **Quant:** `quant_method: fp8`, `activation_scheme: static`, `lm_head`
  excluded.
- **Weights:** ~16 GiB.
- **Context:** native `max_position_embeddings` **163840**, with YaRN
  **already in the checkpoint config** (`rope_scaling` type `yarn`,
  `factor 40`, `original_max_position_embeddings 4096`,
  `mscale`/`mscale_all_dim` 0.707). No override flag needed — just
  `max_model_len ≤ 163840`. `model.env` ships 131072 ("wie aktueller
  Kontext", matches qwen3.8).
- **No MTP head** — MTP/speculative decoding arrived with DeepSeek-V3.
- **License:** DeepSeek license (`license: other` on HF) — permissive,
  commercial use allowed; keep the license file with any redistribution.

## vLLM specifics (this box: NGC `26.07-py3`, vLLM 0.24.0)

- **`VLLM_USE_DEEP_GEMM=0`** applies (box-wide sm_121 assert) — also rules out
  `--moe-backend deep_gemm`.
- **Stock NGC image** — no xgrammar bump needed (that's a tool-calling fix;
  see below).
- **`--trust-remote-code` must be OFF** (`TRUST_REMOTE_CODE=0` in `model.env`).
  The checkpoint's `config.json` `auto_map` points at
  `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`'s custom modeling code;
  with `--trust-remote-code` + `HF_HUB_OFFLINE=1`, `AutoConfig.from_pretrained`
  hard-fails fetching `configuration_deepseek.py` (verified on the first
  start attempt, 2026-09-01). vLLM's native `deepseek_v2` class needs none
  of it. `lib.sh` gained a `TRUST_REMOTE_CODE` knob for this.
- **Do NOT pass `--quantization`** — read from `quantization_config`.
- **Corrected tokenizer REQUIRED** (`--tokenizer /hf-cache/tokenizer-fixes/deepseek-coder-v2-lite`,
  wired by `model.env`). The FP8 repo's `tokenizer_config.json` sets
  `tokenizer_class: "LlamaTokenizer"`; transformers 5.6.1 loads that as
  `LlamaTokenizerFast`, which **overwrites `tokenizer.json`'s ByteLevel
  decoder** with the Llama metaspace decoder (`Replace "▁" -> " "`). DeepSeek's
  vocab uses GPT-2 byte markers (`Ġ`/`Ċ`), so every space and newline is
  dropped from the output — `"Helloworlddeffoo():"`, verified 2026-09-01
  standalone (`AutoTokenizer` on both the FP8 repo AND the original
  `deepseek-ai` repo) and via vLLM. Fix: `./tokenizer/` holds a corrected
  `tokenizer_config.json` (`tokenizer_class: "PreTrainedTokenizerFast"` →
  respects the embedded ByteLevel decoder) + the original `tokenizer.json`;
  `model.env` stages it into the mounted HF cache on first start. `decode()`
  then returns `"Hello world\n  def foo():"` and vLLM output is clean.
- **MLA KV dtype: `auto` (bf16) — the only option on this box.** Both FP8
  KV paths fail:
  - `fp8_ds_mla`: `ValueError: No valid attention backend found for cuda ...
    use_mla=True. Reasons: {TRITON_MLA: [kv_cache_dtype not supported],
    FLASHINFER_MLA_SPARSE_SM120: [requires index_topk config]}` (2026-09-01).
  - `fp8`: `triton.runtime.errors.OutOfResources: out of resource: shared
    memory, Required: 102400, Hardware limit: 101376` — the Triton MLA fp8
    kernel wants 100 KB shared mem, SM121 has 99 KB (2026-09-02 sweep).
  `TRITON_MLA` is the only MLA attention backend here and it needs bf16 KV.
  MLA's KV latent is tiny (~576 elems/token) so fp8 KV would save little
  anyway. `mla_kv_check` keeps `kv_fp8` only to catch a future image that
  lifts the shared-mem limit.
- **`--moe-backend auto`** (picks `FLASHINFER_CUTLASS`). Swept 2026-09-02:
  `auto` == forced `triton` (68.6/171.7 vs 68.9/168.2 tok/s single/batch8 —
  run-to-run noise). Forced `--moe-backend cutlass` -> `ValueError: vLLM
  CUTLASS FP8 MoE backend is disabled for this configuration`. `deep_gemm`
  box-blocked. `EXTRA_VLLM_ARGS` stays tokenizer-only.
- **Tool calling — NOT SUPPORTED, confirmed from the checkpoint.** Its
  `chat_template` (tokenizer_config.json) renders only user/assistant/system
  turns as plain `User: … Assistant: …` — no `tools` block, no tool_calls
  markup. `--tool-call-parser` only parses model *output*; it can't inject
  tool definitions into a prompt the template ignores. `TOOL_CALL_PARSER`
  stays empty → `lib.sh` omits `--enable-auto-tool-choice`. Enabling tool
  calling would require a custom `--chat-template` with tool support (out of
  scope — this is the "fast code completion / chat" model).
- **Reasoning:** not a thinking model — `REASONING_PARSER` empty.

## Resolved

1. **HF_HUB_OFFLINE / auto_map** — `TRUST_REMOTE_CODE=0`.
2. **`fp8_ds_mla` + `fp8` KV both fail on SM121** — default `auto`.
3. **Broken tokenizer (dropped spaces)** — corrected `./tokenizer/`.
4. **MoE backend** — `auto` == `triton`; `cutlass` disabled. Kept `auto`.
5. **`max_num_seqs`** — sweet spot 8, no regression past it.
6. **Prefix caching** — big win, kept on.
7. **Co-residency** — no, does not fit alongside qwen3.8. On-demand only.
8. **Tool calling** — not supported by the checkpoint's chat template.

## Sources

- Model card: <https://huggingface.co/RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8>
- Base model: <https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct>
- <https://docs.vllm.ai/en/stable/features/tool_calling/>
- This box's image queried directly: `docker run --gpus all --entrypoint vllm
  vllm-qwen:26.07-xgrammar024 serve --help=all`
