# RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8 — notes

**Status (2026-09-01): scaffolded + smoke-tested, NOT yet benchmark-tuned.**
Weights downloaded, serves cleanly (health OK, coherent chat, clean
detokenization after the tokenizer fix below, 48k-token prompt handled in
~13 s). Every numeric knob in `model.env` is a reasoned starting point, not a
sweep result — run `bench/matrix.yaml`'s `overnight` suite and fold the
winners in. Three start-blocking issues were found and fixed during the smoke
test — see below (trust-remote-code, `fp8_ds_mla`, tokenizer).

Role on this box: the **performance** coding model (the **quality** one is
`qwen2.5-coder-32b-fp8`). On-demand only — **not** a systemd service. Small
enough (~16 GiB weights + cheap MLA KV) that it *may* fit alongside a live
`qwen3.8-27b-fp8` + `bge-m3` — the sweep's memory numbers decide; document the
verdict here afterwards.

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
- **MLA KV dtype: `auto` (bf16).** `fp8_ds_mla` (the DeepSeek MLA-specific FP8
  KV in the `--kv-cache-dtype` enum) **fails on this box** — verified
  2026-09-01: `ValueError: No valid attention backend found for cuda ...
  kv_cache_dtype=fp8_ds_mla, use_mla=True. Reasons: {TRITON_MLA:
  [kv_cache_dtype not supported], FLASHINFER_MLA_SPARSE_SM120: [requires
  index_topk config]}`. `TRITON_MLA` is the only working MLA attention
  backend in this NGC image on SM121, and it wants bf16 KV. MLA's KV latent
  is ~576 elems/token anyway, so fp8 KV would save little. `mla_kv_check`
  still A/Bs `auto` vs plain `fp8`.
- **`--moe-backend`** is a real axis (MoE model). `moe_backend_probe` tries
  `auto` / `triton` / `cutlass`. Unlike gpt-oss's MXFP4 checkpoint (which
  the flashinfer CUTLASS kernel rejected for its scale-group layout), this is
  a plain FP8 block-quant MoE — `cutlass` may actually work here. **Correctness
  must be curl-verified per backend** (`vllm bench serve` only counts tokens).
- **Tool calling — NOT SUPPORTED, confirmed from the checkpoint.** Its
  `chat_template` (tokenizer_config.json) renders only user/assistant/system
  turns as plain `User: … Assistant: …` — no `tools` block, no tool_calls
  markup. `--tool-call-parser` only parses model *output*; it can't inject
  tool definitions into a prompt the template ignores. `TOOL_CALL_PARSER`
  stays empty → `lib.sh` omits `--enable-auto-tool-choice`. Enabling tool
  calling would require a custom `--chat-template` with tool support (out of
  scope — this is the "fast code completion / chat" model).
- **Reasoning:** not a thinking model — `REASONING_PARSER` empty.

## Open questions for the sweep / first start

1. **RESOLVED: HF_HUB_OFFLINE / auto_map** — `TRUST_REMOTE_CODE=0`, see above.
2. **RESOLVED: `fp8_ds_mla` KV fails on SM121** — default `auto`, see above.
3. **RESOLVED: broken tokenizer (dropped spaces)** — corrected `./tokenizer/`, see above.
4. **MoE backend** — `auto` (picks FLASHINFER_CUTLASS) vs `triton` vs
   `cutlass`, correctness-gated.
5. **`max_num_seqs`** — MoE models often regress past the sweet spot
   (gpt-oss did). `seqs_sweep` 1/2/4/8/16.
6. **Prefix caching** win on coding-shaped traffic.
7. **Co-residency** — does `GPU_MEM_UTIL 0.30` + live qwen3.8 (0.65) + bge-m3
   (0.12) + host fit? First start showed ~19.3 GiB KV pool / 665k tokens /
   5.07x @ 131k at 0.30 solo. Read `mem_ready_mb` / `server_kv_cache_gib`
   from the sweep and record the answer here.

## Sources

- Model card: <https://huggingface.co/RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8>
- Base model: <https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct>
- <https://docs.vllm.ai/en/stable/features/tool_calling/>
- This box's image queried directly: `docker run --gpus all --entrypoint vllm
  vllm-qwen:26.07-xgrammar024 serve --help=all`
