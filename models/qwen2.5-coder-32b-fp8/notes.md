# RedHatAI/Qwen2.5-Coder-32B-Instruct-FP8-dynamic — notes

**Status (2026-09-01): scaffolded + smoke-tested, NOT yet benchmark-tuned.**
Weights downloaded, serves cleanly. Smoke results:
- health OK; `/v1/models` reports `max_model_len 131072`.
- **YaRN works** (not just "doesn't 400"): a needle at line 4321 of a
  106,939-token prompt was retrieved correctly. `--hf-overrides` accepted
  (`hf_overrides: {rope_scaling: {rope_type: yarn, factor: 4.0, ...}}`).
- Clean code output (Rust prime sieve, correct).
- Startup: weights 32.2 GiB / 235 s, torch.compile 25.9 s. **KV pool
  39.34 GiB / 322,288 tokens / 2.46x concurrency @ 131072** at
  `GPU_MEM_UTIL 0.62` — i.e. ~16 GiB per full-length sequence (the
  full-attention KV cost predicted below). Attention backend: **FLASHINFER**
  (works here, unlike gpt-oss); FP8 kernel `CutlassFP8ScaledMMLinearKernel`.
- **Tool calling is unreliable with `tool_choice: "auto"`** — see below.

Every numeric knob in `model.env` is still a reasoned starting point, not a
sweep result — run `bench/matrix.yaml`'s `overnight` suite and fold the
winners in (same process as `gpt-oss-120b` / `qwen3.8-27b-fp8` went through).

Role on this box: the **quality** coding model (the **performance** one is
`deepseek-coder-v2-lite-fp8`). On-demand only — **not** a systemd service;
`qwen3.8-27b-fp8` must be stopped first (memory). See the repo README.

## What it is

- **Base:** `Qwen/Qwen2.5-Coder-32B-Instruct` (Apache 2.0), offline FP8-quantized
  by RedHat/Neural Magic with `llm-compressor`.
- **Architecture:** `Qwen2ForCausalLM` — **dense**, 64 layers, hidden 5120,
  40 attention heads / 8 KV heads, head_dim 128, `intermediate_size` 27648.
  **Full attention on every layer** (no sliding window, no linear attention).
- **Quant:** `compressed-tensors`, `format: float-quantized` — **W8A8**:
  per-channel FP8 weights (static, mse observer) + **dynamic per-token** FP8
  activations. `lm_head` excluded (stays BF16). `kv_cache_scheme: null`
  (no k/v_scale in the checkpoint).
- **Weights:** ~33–34 GiB.
- **Context:** native `max_position_embeddings` **32768**, `rope_scaling: null`.
  Qwen documents YaRN `factor 4.0` → **131072** as the supported extension.
- **License:** Apache 2.0.

## vLLM specifics (this box: NGC `26.07-py3`, vLLM 0.24.0)

- **`VLLM_USE_DEEP_GEMM=0`** still applies (box.env, box-wide sm_121 assert).
- **Do NOT pass `--quantization`** — read from `quantization_config`. Same rule
  as qwen3.8's fp8 / gpt-oss's mxfp4.
- **No `--rope-scaling` flag in this build** (checked: `vllm serve --help=all`
  lists only `--hf-overrides` / `--rope-theta`). YaRN goes through
  `--hf-overrides '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'`,
  wired via `model.env`'s `EXTRA_VLLM_ARGS` (compact JSON, single argv token).
- **Tool calling** needs xgrammar ≥ 0.2.4 — same import bug + fix as qwen3.8
  (`cannot import name 'normalize_tool_choice'`). `model.env` reuses the
  `vllm-qwen:26.07-xgrammar024` image. **Not registry-pullable** — rebuild
  from `../qwen3.8-27b-fp8/docker/Dockerfile.xgrammar024` if the local image
  cache loses it.
- **`--tool-call-parser hermes`** — Qwen2.5's documented function-calling
  format, present in this image's parser enum. Not a thinking model →
  `REASONING_PARSER` stays empty (`lib.sh` omits `--reasoning-parser`).
- **Tool calling caveat (verified 2026-09-01).** The chat template is
  correct (renders `<tools>` for signatures, instructs `<tool_call>…</tool_call>`
  for calls) and `hermes` is the right parser — but with
  `tool_choice: "auto"` this 32B Coder **consistently emits the wrong
  wrapper** (`<tools>{json}</tools>` — echoing the signature tag) instead of
  `<tool_call>…`, so `hermes` finds nothing and the call falls through as
  `content` (3/3 fails at temp 0 and 0.7). With `tool_choice: "required"`
  or a named function it **works** (guided decoding forces valid
  `tool_calls`). This is a known Qwen2.5-**Coder** quirk (the code
  fine-tune degraded tool-format adherence vs Qwen2.5-Instruct), not a
  serving-config bug. Clients that need reliable tool use here should pass
  `tool_choice: "required"`/named when a tool is expected; free-form
  `auto` tool use is not dependable. Pure code-gen (the main use) is
  unaffected.
- **KV per token ≈ 128 KiB** at fp8 KV (`2·64·8·128·1 B`) → **~16 GiB for one
  full-length 131072 sequence**. Full-attention math — very different from
  qwen3.8's hybrid stack (which fit ~1M tokens in ~45 GiB). `GPU_MEM_UTIL`
  and the context-scaling sweep rows account for this.

## Open questions for the sweep / first start

1. **RESOLVED: YaRN works** — 106,939-token needle test passed;
   `rope_type: yarn` is the accepted key.
2. **`yarn_cost` suite** — how much does static YaRN (factor 4, always on)
   cost on < 32k prompts vs. a native-32768 server? If large, consider
   shipping native 32768 + a documented one-off env var for the rare long job.
3. **KV dtype** — `fp8` vs `auto` (`kv_sweep`); the checkpoint has no
   k/v_scale (accuracy caveat, unmeasured).
4. **`max_num_seqs`** sweet spot (`seqs_sweep`) — expect low (on-demand model).
5. **Prefix caching** win size on coding-shaped traffic (`prefix_cache_ab`).
6. **`ctx_256k_probe`** — factor 8 YaRN, untested by Qwen. Data point only;
   adopt only if KV pool / TTFT / correctness all hold up.
7. **`GPU_MEM_UTIL`** — derive from `server_kv_cache_gib` / `mem_ready_mb`;
   the 32B + 16 GiB KV sequence likely wants more than the 0.62 placeholder.

## Sources

- Model card: <https://huggingface.co/RedHatAI/Qwen2.5-Coder-32B-Instruct-FP8-dynamic>
- Base model + YaRN guidance: <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct>
- <https://qwen.readthedocs.io/en/latest/deployment/vllm.html>
- This box's image queried directly: `docker run --gpus all --entrypoint vllm
  vllm-qwen:26.07-xgrammar024 serve --help=all`
