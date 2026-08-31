# BAAI/bge-m3 — notes

## What it is

- **Task:** multilingual text embeddings. Served here as **dense** vectors over
  the OpenAI-compatible `POST /v1/embeddings` (model id `bge-m3`).
- **Architecture:** `XLMRobertaModel` encoder (BERT-family, bidirectional),
  ~568M params, hidden size 1024. Position embeddings extended to 8194 →
  **`--max-model-len 8192`**.
- **Output:** 1024-dim dense embedding, **CLS pooling + L2-normalized**
  (`1_Pooling/config.json` = CLS; `sentence_bert_config.json` /
  `config_sentence_transformers.json` = normalize). bge-m3 is "M3" =
  Multi-Functionality (dense + sparse + ColBERT/multi-vector),
  Multi-Linguality (100+ languages), Multi-Granularity (up to 8192 tokens).
  **vLLM's `/v1/embeddings` serves only the dense head.** The sparse
  (`sparse_linear.pt`) and ColBERT (`colbert_linear.pt`) heads are in the
  checkpoint but not exposed by this endpoint — fine for our use (dense
  retrieval / RAG similarity via the litellm proxy).
- **License:** MIT.

## vLLM specifics (0.24.0 / NGC 26.07, sm_121 / GB10)

- **`--task` was removed in 0.24.0.** The embedding mode is
  **`--runner pooling`** (`--runner {auto,draft,generate,pooling}`). `--convert
  embed` is only for adapting a *generative* checkpoint; bge-m3 is a native
  pooling model, so `auto` detection under `--runner pooling` is enough.
- **Pooling + normalization are read from the model config** — do NOT pass
  `--pooler-config`. If a smoke test shows un-normalized vectors (‖v‖ ≠ 1) or a
  wrong dim, only then override.
- **No `--trust-remote-code`** — XLM-RoBERTa is natively supported.
- **`--dtype bfloat16`** — the checkpoint is fp32; bf16 is the GB10 native
  compute dtype and halves the footprint with no measurable quality cost.
- **`VLLM_USE_DEEP_GEMM=0`** (box.env) is injected as always but is a no-op here
  (no FP8 path in a bf16 encoder).
- Runs from the **stock `nvcr.io/nvidia/vllm:26.07-py3`** image — no derived
  image needed (the `vllm-qwen:26.07-xgrammar024` bump is a tool-calling fix).

### If `--runner pooling` fails on this build / sm_121

The encoder path is plain bidirectional attention and should be fine, but if the
pooling runner errors at load or returns garbage:

1. Try `EXTRA_VLLM_ARGS="--convert embed"`.
2. Fall back off vLLM: **HF `text-embeddings-inference`** (TEI) or **Infinity**
   or **sentence-transformers** in a small container on the same port/#auth.
   BAAI publishes bge-m3 for all three. Keep the same `model.env` /
   systemd-instance shape; only `serve.sh`'s launch line changes.

## Serving footprint

- `GPU_MEM_UTIL=0.12` (~15 GB ceiling of the 128 GB unified pool) — a generous
  cap, not a target (~1.1 GiB bf16 weights + activations). Sits next to
  `qwen3.8-27b-fp8` at 0.65 → sum 0.77, ~0.23 pool left for the host.
- `STARTUP_TIMEOUT=1200` **must** equal qwen's — `install-service` rewrites the
  shared `llm-vllm@.service` template with this value (see `model.env`).

## Multilingual sanity check

A German and an English sentence with the same meaning should have high cosine
similarity (bge-m3 is trained for cross-lingual retrieval), clearly above an
unrelated pair. Example: `"Der schnelle braune Fuchs springt über den faulen
Hund"` vs `"The quick brown fox jumps over the lazy dog"`.

## Download

```bash
HF_HOME=/home/ss/models/hf-cache /home/ss/llm/.hf-venv/bin/hf download BAAI/bge-m3
```

Public repo, no `HF_TOKEN`. The container runs `HF_HUB_OFFLINE=1`, so the
weights must be in `/home/ss/models/hf-cache/hub/models--BAAI--bge-m3` before
the first start (`preflight()` in `lib.sh` checks and aborts with this hint).

## Sources

- Model card: <https://huggingface.co/BAAI/bge-m3>
- Paper (M3-Embedding, BGE-M3): arXiv:2402.03216
- vLLM pooling models: <https://docs.vllm.ai/en/latest/models/pooling_models.html>
