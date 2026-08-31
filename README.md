# LLM serving on the DGX Spark (GB10)

vLLM serving + benchmark harness for LLMs on a single **NVIDIA GB10**
(DGX Spark, 128 GB unified LPDDR5x, sm_121, CUDA 13).

One generic launcher in `common/`, one directory per model in `models/`.
Adding a model = a new `models/<slug>/` with a `model.env`.

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
│   └── bench/
│       ├── matrix.yaml   the sweep (configs, workloads, suites)
│       ├── ocr_test.png
│       └── results/      per-run JSON, server logs, summary.csv, report.md/html, plots
└── gpt-oss-120b/       native MXFP4 text MoE, Harmony format   (scaffolded; benchmark pending)
.env                    box-wide secrets: API_KEY, HF_TOKEN
.hf-venv/               venv for the bench harness + hf downloads
```

Weights and the vLLM compile cache live outside the repo:
`/home/ss/models/hf-cache`, `/home/ss/models/vllm-cache`.

## Config layers

`common/box.env` (hardware) → `models/<slug>/model.env` (the model) →
a real environment variable at call time (`MAX_MODEL_LEN=32768 ./serve start`).
Each `.env` file uses `: "${K:=default}"`, so a set env var always wins.

`EXTRA_VLLM_ARGS` (a plain space-separated string) is appended verbatim to
`vllm serve` — the escape hatch for flags with no dedicated knob, e.g.
`EXTRA_VLLM_ARGS="--moe-backend marlin --attention-backend TRITON_ATTN" ./serve start`.
Empty by default; `gpt-oss-120b/model.env` documents the SM121 MXFP4 backend
overrides it exists for.

---

## Serve a model

```bash
cd models/qwen3.8-27b-fp8

./serve                 # start: detached, wait for /health, print the API URL
./serve status          # systemd state + container health + API URLs + model list
./serve logs
./serve stop
```

Prerequisites: Docker + NVIDIA Container Toolkit; `.env` with `API_KEY`
(`openssl rand -hex 32`) and `HF_TOKEN`; the weights downloaded —

```bash
HF_HOME=/home/ss/models/hf-cache HF_TOKEN=hf_... \
  .hf-venv/bin/hf download orcarouter/Qwen3.8-27B-Uncensored-FP8
```

First start ~7 min (weight load + torch.compile + CUDA-graph + vision encoder);
later starts are faster (compile cache persists). It prints:

```
  base URL (local)  : http://localhost:8000/v1
  base URL (network): http://<lan-ip>:8000/v1
  model id          : qwen3.8-27b-uncensored
  auth              : header  'Authorization: Bearer <API_KEY>'   (required)
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"qwen3.8-27b-uncensored","messages":[{"role":"user","content":"hi"}]}'
```

- **No thinking:** add `"chat_template_kwargs": {"enable_thinking": false}`.
- **Vision / OCR:** `content` = a list of a `text` part and an `image_url` part
  (`data:image/png;base64,...`).

## Persistent across reboots (systemd)

```bash
./serve install-service     # generates + enables  llm-vllm@qwen3.8-27b-fp8  (sudo)

systemctl status llm-vllm@qwen3.8-27b-fp8
journalctl -u  llm-vllm@qwen3.8-27b-fp8 -f
sudo systemctl restart llm-vllm@qwen3.8-27b-fp8   # returns once serving again
./serve uninstall-service
```

One template unit, one instance per model slug. systemd is the single
supervisor — in this mode `serve.sh ... run` execs `docker run --rm` in the
foreground, so no container `--restart` policy races with it on boot. If the
unit is installed, use `systemctl`, not the bare `start`/`stop`.

---

## Running multiple models

Each `model.env` sets its own `PORT` and `GPU_MEM_UTIL`. The 128 GB unified
pool is shared: **the sum of every running model's `GPU_MEM_UTIL` plus host
headroom must stay under 1.0.** e.g. two models at `0.40` each + `~0.15` host.
Container names are `vllm-<slug>`, benchmark containers `bench-<slug>`, so they
never collide.

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

---

## Box-wide gotchas (`common/box.env`)

- **`VLLM_USE_DEEP_GEMM=0` is mandatory** on sm_121 — DeepGEMM's UE8M0 FP8 path
  asserts `Unknown recipe` during kernel warmup.
- **Unified memory:** `--gpu-memory-utilization` carves from the whole 128 GB
  pool. `nvidia-smi` reports `N/A` for GPU memory — use `free -m`.
- vLLM 0.24.0: MTP is `--speculative-config`, not `--num-speculative-tokens`.
- Models too big for one GB10 stay out (`Qwen3.8-Flash-Next`: `qwen4_exp`,
  ~180B MoE, ~186 GB in any quant, needs TP ≥ 8).

## License

Per-model — see each `models/<slug>/notes.md`. `qwen3.8-27b-fp8` is Apache 2.0,
**uncensored / abliterated** (no built-in guardrails; research use, add your own
moderation). `gpt-oss-120b` is Apache 2.0 (OpenAI, stock — keeps its own
safety training). Scripts in this repo: MIT.
