# gemma4-26b-heretic-nvfp4 — notes

`llmfan46/gemma-4-26B-A4B-it-qat-q4_0-uncensored-heretic-NVFP4`, served as
**gb10-writing** (port 8004) — German stories and interactive roleplay,
uncensored. Runs **in parallel** with gb10-reasoning (qwen3.8-uc at 0.32),
gb10-embedding and gb10-rendering (ComfyUI).

## Why this model (2026-10-10)

qwen3.8 reasons well but writes weak German prose and, worse, collapses in
long roleplay: once a phrase is in context it repeats it, and from turn ~6
of a 12-turn RP it degenerates (reproducible, see the qwen notes). Mistral
Small 4 Heretic writes the cleanest single texts but needs ~66 GiB weights
— never co-resident with qwen. The user wanted a parallel writer instead of
switching, **uncensored only** (abliterated / Heretic variants; refusal test
is pass/fail).

Gemma 4 26B-A4B: MoE, 128 experts top-8, ~4B active → ~30 tok/s on GB10.
25 sliding-window (1024) + 5 full-attention layers with 2 KV heads → ~20 KB
KV per token, so long RP sessions are cheap. Heretic v1.2 (ARA on
attn.o_proj; card: 11/100 refusals, KL 0.06), NVFP4 by ModelOpt.

Small 4 Heretic was tested as an on-demand gb10-reasoning member and then
**removed** (2026-10-10, user decision: not co-resident, rejects the agent harness's
`chat_template_kwargs` with HTTP 400). Its recipe (native-format
conversion, model.env, notes) is in git history at `1d128a3`
(`models/mistral-small4-heretic/`).

## Startup (vLLM 0.24, GPU_MEM_UTIL 0.22)

- 18.13 GiB weights, 4.9 GiB KV = **202,316 tokens** (MAX_MODEL_LEN 131072,
  MAX_NUM_SEQS 8 — plenty).
- **Stock 26.07 image crashes at load**: `NotImplementedError` in
  `tie_weights`. The checkpoint excludes the tied `lm_head` from
  quantization; vLLM 0.24's ModelOpt config hands an excluded
  `ParallelLMHead` an `UnquantizedLinearMethod`, which has no
  `tie_weights` (a secondary UnicodeDecodeError follows). Fix: return
  `UnquantizedEmbeddingMethod` for `ParallelLMHead` —
  `docker/patch_modelopt_tied_head.py`, baked into
  `vllm-gb10:26.07-xg024-tiedhead` (on top of `vllm-qwen:26.07-xgrammar024`,
  so the xgrammar 0.2.4 tool-call fix is included). Build:
  ```bash
  docker build -t vllm-gb10:26.07-xg024-tiedhead \
    -f models/gemma4-26b-heretic-nvfp4/docker/Dockerfile.tiedhead \
    models/gemma4-26b-heretic-nvfp4/docker
  ```
  `model.env` swaps the image only when the box default is the stock 26.07.
  Drop the override once upstream vLLM handles excluded tied heads.
- Download: plain HF layout via curl into `${HF_CACHE}/local/...`, SHA256
  checked against the LFS OIDs, no `.py` files in the repo.

## Thinking

**On by default** (`--default-chat-template-kwargs {"enable_thinking":true}`
in model.env; the user wants the visible reasoning). Per request off:
`chat_template_kwargs: {"enable_thinking": false}`.

Known issue (2026-10-10): on a long agent-harness session (~88k tokens) the
thinking phase never closed. The model wrote "(End of thought)" /
"(End of thought process)" and then "(End)" lines endlessly *inside the
reasoning channel* (the client showed them as reasoning), with the GPU busy
until max_tokens. It happened with the server default on and also with it
off (the client sends thinking itself). Synthetic 90k histories without
thinking did not reproduce it with any sampling setting tried. The
user started a fresh session. If it recurs, the guard is vLLM's
`thinking_token_budget` (per request) + `--reasoning-config` with
Gemma's markers (`<|channel>` / `<channel|>`). That needs the V1 model
runner and is not set up yet. The client doesn't abort the request on Stop;
only a service restart kills a runaway.

## Context 256k (2026-10-10)

MAX_MODEL_LEN 131072 -> 262144 (native max). 0.24 failed at startup (needs
5.67 GiB KV for one full sequence, got 5.17); 0.26 gave 13.75 GiB KV =
635k tokens. No long-context quality test run (user: not needed).

## Test results (2026-10-10, same prompts for all models)

| | qwen3.8-uc | Small 4 Heretic | **Gemma 4 26B Heretic** |
|---|---|---|---|
| tok/s single stream | 11–16.5 | ~31 | ~30 |
| 3-part saga, repeated 4-grams / 1000 words | 289 | 33 | **8** |
| 12-turn RP, thinking off | collapses from turn 6 | 12/12, speaks for the user twice | **12/12, ≤4 % reuse, never speaks for the user** |
| 12-turn RP, thinking on | 11/12 (turn 11 empty), 78 s/turn | – | 12/12, ~1 % reuse, 38 s/turn |
| short story (~800 words) | many grammar errors | cleanest | some grammar errors, "not X, but Y" construction tic |
| agent-style tool loop (fake NHL data) | 4/4, 16 s | 4/4, 7 s | 4/4, 8 s |
| code (3 tasks, hidden asserts), think off / on | 3/3 / – | 2/3 / – | 3/3 / 3/3 |
| refusals (4 explicit/dark prompts) | 0/4 | 0/4 | 0/4 |

Verdict: best of the three for long stories and roleplay (the actual
problem), fast, fits in parallel. Single texts are less polished than
Small 4. Also a good fit for tool-using cron jobs (2× qwen's speed, takes
load off qwen).

## Open

- Image input not tested yet (vision tower loaded, MM_IMAGE_LIMIT 1).
- FLUX render while qwen + Gemma + bge run: not yet tested under load.
