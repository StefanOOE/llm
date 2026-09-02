# LLM serving on the DGX Spark (GB10)

vLLM serving + benchmark harness for LLMs on a single **NVIDIA GB10**
(DGX Spark, 128 GB unified LPDDR5x, sm_121, CUDA 13).

One generic launcher in `common/`, one directory per model in `models/`.
Adding a model = a new `models/<slug>/` with a `model.env` (an embedding model
just adds `RUNNER=pooling` — see `bge-m3/`).

```
common/
├── box.env             hardware facts: NGC image, caches, sm_121 quirks
├── serve.sh            generic launcher   (serve.sh <model-dir> <subcommand>)
├── lib.sh              shared implementation
├── llm-vllm@.service   systemd template (reference; install-service generates the real one)
└── bench/              run.py · report.py · ab_lmonly.py · _env.py
models/
├── qwen3.8-27b-fp8/    FP8 vision MoE, MTP speculative decoding   (benchmark-tuned)
│   ├── model.env       identity + benchmark-tuned serving knobs   ← the parametrization
│   ├── notes.md        model-card facts + gotchas
│   ├── serve           thin wrapper -> common/serve.sh with this dir
│   ├── docker/         Dockerfile.xgrammar024 (xgrammar 0.2.0->0.2.4 patch, see notes.md)
│   └── bench/
│       ├── matrix.yaml   the sweep (configs, workloads, suites)
│       ├── ocr_test.png
│       └── results/      per-run JSON, server logs, summary.csv, report.md/html, plots
├── gpt-oss-120b/       native MXFP4 text MoE, Harmony format   (benchmark-tuned)
│   ├── model.env       same layout as above, all knobs benchmark-tuned
│   ├── notes.md         model-card facts, SM121 MXFP4 bug (verified NOT reproducing), sweep results
│   ├── serve            thin wrapper -> common/serve.sh with this dir
│   └── bench/
│       ├── matrix.yaml   the sweep (configs, workloads, suites)
│       └── results/      per-run JSON, server logs, summary.csv (4 suites: backend/seqs/kv/context)
├── qwen3-coder-30b-a3b-fp8/  MoE block-FP8 coder, 131072 ctx   (quality coder; benchmark-tuned)
│   ├── model.env           benchmark-tuned knobs; qwen3_coder tool parser
│   ├── notes.md            why not Qwen2.5-Coder-32B (bandwidth wall), MoE facts, sweep results
│   ├── serve               thin wrapper -> common/serve.sh with this dir
│   └── bench/matrix.yaml    moe_backend / kv / seqs / prefix / context suites
├── deepseek-coder-v2-lite-fp8/  MoE+MLA FP8 coder, 131072 ctx   (performance coder; benchmark-tuned)
│   ├── model.env           benchmark-tuned knobs; KV auto (fp8 fails on SM121 MLA), corrected tokenizer
│   ├── notes.md            MoE/MLA facts, 3 fixed start-blockers, sweep results
│   ├── serve               thin wrapper -> common/serve.sh with this dir
│   ├── tokenizer/          corrected tokenizer_config.json (FP8 repo's breaks the ByteLevel decoder; model.env stages it + tokenizer.json from the snapshot)
│   └── bench/matrix.yaml    moe_backend / mla_kv / seqs / prefix / context suites
└── bge-m3/            XLM-RoBERTa encoder, 1024-dim multilingual embeddings   (RUNNER=pooling)
    ├── model.env       identity + pooling knobs (no chat/generate flags)
    ├── notes.md         dense/sparse/ColBERT facts, --runner pooling, fallbacks
    └── serve            thin wrapper -> common/serve.sh with this dir
llmctl                  interactive terminal dashboard: toggle models, memory-fit check
.env                    box-wide secrets: API_KEY, HF_TOKEN
.hf-venv/               venv for the bench harness + hf downloads
```

Weights and the vLLM compile cache live outside the repo:
`/home/ss/models/hf-cache`, `/home/ss/models/vllm-cache`.

## Models on this box

| slug | served as | port | endpoint | context | license | status |
|---|---|---|---|---|---|---|
| `qwen3.8-27b-fp8` | `qwen3.8-27b-uncensored` | 8000 | `/v1/chat/completions` | 131072 | Apache 2.0, uncensored | benchmark-tuned, in daily use (systemd) |
| `bge-m3` | `bge-m3` | 8001 | `/v1/embeddings` | 8192 | MIT | in use, embeddings for the litellm proxy (systemd) |
| `gpt-oss-120b` | `gpt-oss-120b` | 8002 | `/v1/chat/completions` | 65536 | Apache 2.0, stock safety | benchmark-tuned, on-demand |
| `qwen3-coder-30b-a3b-fp8` | `qwen3-coder-30b` | 8003 | `/v1/chat/completions` | 131072 | Apache 2.0 | quality coder — benchmark-tuned, on-demand |
| `deepseek-coder-v2-lite-fp8` | `deepseek-coder-v2-lite` | 8004 | `/v1/chat/completions` | 131072 | DeepSeek license | performance coder — benchmark-tuned, on-demand |

All share the 128 GB unified pool — see **Running multiple models** below.
**Fixed (systemd, boot-persistent):** `qwen3.8-27b-fp8` (`0.65`) + `bge-m3`
(`0.12`). **On-demand** (`./serve start` / `stop`, no service): `gpt-oss-120b`,
`qwen3-coder-30b-a3b-fp8`, `deepseek-coder-v2-lite-fp8` — each needs
`qwen3.8-27b-fp8` stopped first for memory (confirmed by each model's sweep,
`deepseek-coder-v2-lite` included).

## Config layers

`common/box.env` (hardware) → `models/<slug>/model.env` (the model) →
a real environment variable at call time (`MAX_MODEL_LEN=32768 ./serve start`).
Each `.env` file uses `: "${K:=default}"`, so a set env var always wins.

`EXTRA_VLLM_ARGS` (a plain space-separated string) is appended verbatim to
`vllm serve` — the escape hatch for flags with no dedicated knob, e.g.
`EXTRA_VLLM_ARGS="--moe-backend marlin --attention-backend TRITON_ATTN" ./serve start`.
Empty by default; `gpt-oss-120b/model.env` documents the SM121 MXFP4 backend
overrides it exists for.

`RUNNER` (default `generate`) picks the vLLM 0.24.0 runner. `generate` gets the
full chat flag set (parsers, tool-choice, kv-cache-dtype, MTP, mm-limits).
`pooling` is for embedding models served at `/v1/embeddings` — a minimal flag
set with none of those; it also reads `DTYPE` (passed as `--dtype` when not
`auto`) and `ENFORCE_EAGER` (`1` adds `--enforce-eager`). CLS-pooling and
L2-norm come from the model's own config — see `bge-m3/`.

**Gotcha:** a few vars (`VLLM_IMAGE`, `HF_CACHE`, `VLLM_CACHE`, ...) already
get a value from `box.env`'s own `:=`, so a `model.env` line using `:=` on
one of *those* is a silent no-op (the var is already non-empty by the time
it runs) — a real per-model override needs a plain assignment instead, e.g.
`qwen3.8-27b-fp8/model.env`'s `VLLM_IMAGE` override (only replaces it if
still `box.env`'s stock default, so a one-off env var still wins).

---

## `llmctl` — interactive control

```bash
./llmctl
```

A one-screen terminal dashboard: every model, whether it's running, its pool
reservation (`GPU_MEM_UTIL` from `model.env`) and port. Type a number to
toggle it — running → stop, stopped → start. A start is refused up front if
the running models' `GPU_MEM_UTIL` sum plus the candidate would exceed the
pool budget (`POOL_BUDGET`, default `0.90` — host keeps ~0.10), telling you
which model to stop first. Stopping a fixed systemd model (`qwen3.8-27b-fp8`,
`bge-m3`) asks `y/N` first. `q` or Ctrl-C quits.

It just wraps `models/<slug>/serve start|stop` (on-demand models) and
`sudo systemctl start|stop llm-vllm@<slug>` (systemd models) — same as doing
it by hand, minus the memory arithmetic. The one-line role shown per model
comes from a `# llmctl: …` comment in each `model.env`.

## Serve a model

```bash
cd models/<slug>        # qwen3.8-27b-fp8 | bge-m3 | gpt-oss-120b | qwen3-coder-30b-a3b-fp8 | deepseek-coder-v2-lite-fp8

./serve                 # start: detached, wait for /health, print the API URL
./serve status          # systemd state + container health + API URLs + model list
./serve logs
./serve stop
```

Prerequisites: Docker + NVIDIA Container Toolkit; `.env` with `API_KEY`
(`openssl rand -hex 32`) and `HF_TOKEN` (only needed for gated repos); the
weights downloaded —

```bash
# qwen3.8-27b-fp8 (gated -> needs HF_TOKEN)
HF_HOME=/home/ss/models/hf-cache HF_TOKEN=hf_... \
  .hf-venv/bin/hf download orcarouter/Qwen3.8-27B-Uncensored-FP8

# gpt-oss-120b (public, no token; skip the Apple-metal and OpenAI-native
# weight formats -- vLLM only needs the HF-format safetensors shards)
HF_HOME=/home/ss/models/hf-cache \
  .hf-venv/bin/hf download openai/gpt-oss-120b --exclude "metal/*" --exclude "original/*"

# bge-m3 (public, no token)
HF_HOME=/home/ss/models/hf-cache .hf-venv/bin/hf download BAAI/bge-m3

# qwen3-coder-30b-a3b-fp8  /  deepseek-coder-v2-lite-fp8 (both public, no token)
HF_HOME=/home/ss/models/hf-cache .hf-venv/bin/hf download Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
HF_HOME=/home/ss/models/hf-cache .hf-venv/bin/hf download RedHatAI/DeepSeek-Coder-V2-Lite-Instruct-FP8
```

First start is slow (weight load + torch.compile + CUDA-graph, + vision
encoder for qwen, + ~63 GiB cold load for gpt-oss-120b, + ~31 GiB for
qwen3-coder-30b-a3b — `STARTUP_TIMEOUT` in each `model.env` is sized for this);
later starts are faster (compile cache persists). It prints:

```
  base URL (local)  : http://localhost:<port>/v1
  base URL (network): http://<lan-ip>:<port>/v1
  model id          : <served-name>
  auth              : header  'Authorization: Bearer <API_KEY>'   (required)
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"qwen3.8-27b-uncensored","messages":[{"role":"user","content":"hi"}]}'
```

- **No thinking:** add `"chat_template_kwargs": {"enable_thinking": false}`
  (qwen3.8-27b-fp8) — gpt-oss-120b's Harmony reasoning channel comes back in
  the `reasoning` field regardless, not toggleable per request the same way.
- **Vision / OCR (qwen3.8-27b-fp8 only):** `content` = a list of a `text`
  part and an `image_url` part (`data:image/png;base64,...`). gpt-oss-120b is
  text-only.
- **Tool calling:** both models run with `--enable-auto-tool-choice`; qwen
  uses its native `qwen3_coder` parser, gpt-oss-120b the Harmony
  `openai`/`openai_gptoss` parsers (see each `notes.md`).

**Embeddings (`bge-m3`, `RUNNER=pooling`):** served at `/v1/embeddings`, not
chat. 1024-dim dense vectors, CLS-pooled and L2-normalized (from the model
config — nothing to pass).

```bash
curl -s http://localhost:8001/v1/embeddings \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"bge-m3","input":["Hallo Welt","the quick brown fox"]}'
# -> data[0].embedding has length 1024
```

## Persistent across reboots (systemd)

```bash
./serve install-service     # generates + enables  llm-vllm@<slug>  (sudo)

systemctl status llm-vllm@<slug>
journalctl -u  llm-vllm@<slug> -f
sudo systemctl restart llm-vllm@<slug>   # returns once serving again
./serve uninstall-service
```

Currently installed: `llm-vllm@qwen3.8-27b-fp8` and `llm-vllm@bge-m3` (both
enabled, boot automatically). `gpt-oss-120b`, `qwen3-coder-30b-a3b-fp8` and
`deepseek-coder-v2-lite-fp8` are **on-demand by design** — not installed as
services, started ad-hoc via `./serve start` / `./serve stop`.

One template unit, one instance per model slug. systemd is the single
supervisor — in this mode `serve.sh ... run` execs `docker run --rm` in the
foreground, so no container `--restart` policy races with it on boot. If the
unit is installed, use `systemctl`, not the bare `start`/`stop`.

**The `llm-vllm@.service` template is shared and regenerated by every
`install-service`** with that model's `TimeoutStartSec` / `STARTUP_TIMEOUT` —
so `bge-m3/model.env` deliberately keeps `STARTUP_TIMEOUT=1200` (= qwen's) to
leave the installed unit byte-identical.

---

## Running multiple models

Each `model.env` sets its own `PORT` and `GPU_MEM_UTIL`. The 128 GB unified
pool is shared: **the sum of every running model's `GPU_MEM_UTIL` plus host
headroom must stay under 1.0.** e.g. two models at `0.40` each + `~0.15` host.
Container names are `vllm-<slug>`, benchmark containers `bench-<slug>`, so they
never collide.

**Fixed pair (systemd):** qwen (`0.65`) + bge-m3 (`0.12`) = `0.77`, ~0.23 of
the pool left for the host — fits, both boot-persistent.

**On-demand models — all need qwen3.8 stopped first (memory):**
- **`gpt-oss-120b` (`0.85`)** — ~66 GiB weights + ~34-35 GiB KV, ~101 GB of
  the ~109 GB carved.
- **`qwen3-coder-30b-a3b-fp8` (`0.40` placeholder)** — MoE, ~31 GiB weights
  (all experts resident, only compute is sparse) + KV. `GPU_MEM_UTIL` from
  its sweep.
- **`deepseek-coder-v2-lite-fp8` (`0.30`, benchmark-tuned)** — ~16 GiB
  weights + cheap MLA KV, but the sweep showed ~44-52 GiB peak unified-mem
  use → **does not** co-reside with the fixed pair either (`0.30` + qwen's
  ~83 GiB + bge's ~15 GiB ≈ 148 > 128).

bge-m3 is small enough to leave running in every case. Ports are distinct per
model (8000/8001/8002/8003/8004); container names are `vllm-<slug>`, benchmark
containers `bench-<slug>`, so nothing collides.

## Benchmarking

The harness reuses the model's serving port, so **stop the server first**.

```bash
models/<slug>/serve stop
sg docker -c ".hf-venv/bin/python common/bench/run.py --model models/<slug> --suite smoke"
sg docker -c ".hf-venv/bin/python common/bench/run.py --model models/<slug> --suite overnight"
python3 common/bench/report.py models/<slug>/bench/results/<suite>_<ts>
sg docker -c ".hf-venv/bin/python common/bench/ab_lmonly.py --model models/<slug>"
```

See `common/bench/README.md`. For `qwen3.8-27b-fp8` the report is at
`models/qwen3.8-27b-fp8/bench/results/overnight_20260827_191752/report.md`
(and <https://claude.ai/code/artifact/0843d0cb-f656-4f09-8e08-3a6fa499fd67>).

`gpt-oss-120b`'s `bench/matrix.yaml` sweep (backend/seqs/kv/context-scaling)
is done — results in `bench/results/*_20260831_*/`, winning config already
folded into `model.env`. The SM121 MXFP4 "null Harmony token" bug that
motivated the correctness-gating on `backend_*` (see `notes.md`) was
manually curl-verified **not** to reproduce on this box's image.

---

## Box-wide gotchas (`common/box.env`)

- **`VLLM_USE_DEEP_GEMM=0` is mandatory** on sm_121 — DeepGEMM's UE8M0 FP8 path
  asserts `Unknown recipe` during kernel warmup.
- **Unified memory:** `--gpu-memory-utilization` carves from the whole 128 GB
  pool. `nvidia-smi` reports `N/A` for GPU memory — use `free -m`.
- vLLM 0.24.0: MTP is `--speculative-config`, not `--num-speculative-tokens`.
- **NGC `26.07-py3` ships `xgrammar==0.2.0`**, but vLLM 0.24.0's tool-calling
  path needs `normalize_tool_choice` (added in xgrammar 0.2.4) — any request
  with `"tools"` set 500s (`cannot import name 'normalize_tool_choice'`) on
  the stock image. Any model that needs tool calling on this box needs the
  same `--no-deps` xgrammar bump; see
  `models/qwen3.8-27b-fp8/docker/Dockerfile.xgrammar024` for the pattern and
  its `model.env`'s `VLLM_IMAGE` override for how to wire it in.
- Models too big for one GB10 stay out (`Qwen3.8-Flash-Next`: `qwen4_exp`,
  ~180B MoE, ~186 GB in any quant, needs TP ≥ 8).

## License

Per-model — see each `models/<slug>/notes.md`. `qwen3.8-27b-fp8` is Apache 2.0,
**uncensored / abliterated** (no built-in guardrails; research use, add your own
moderation). `gpt-oss-120b` is Apache 2.0 (OpenAI, stock — keeps its own
safety training). `bge-m3` is MIT (BAAI). Scripts in this repo: MIT.
