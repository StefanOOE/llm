#!/usr/bin/env python3
"""convert-flux-diffusers.py -- diffusers-format FLUX transformer (sharded)
-> single ComfyUI/BFL-key safetensors file, linear weights cast to fp8_e4m3fn.

Why: some FLUX.1-dev finetunes on HF only ship the
diffusers layout (3 bf16 shards, ~24 GB). ComfyUI's UNETLoader wants one
file in diffusion_models/; fp8 halves the resident size (~12 GB), which
matters on this box's shared unified-memory pool (see notes.md).

Streams tensor-by-tensor and writes the safetensors file incrementally, so
peak RAM stays at one fused tensor -- a naive load+convert+save would need
~24+12 GB, more than is free while the vLLM models are running.

Uses ComfyUI's own key map (comfy.utils.flux_to_diffusers), so the result
is exactly what ComfyUI's runtime diffusers conversion would produce.
Runs inside the comfyui image:

  docker run --rm -v <src>:/src:ro -v <dst>:/dst -v $PWD:/tools:ro \
    --entrypoint python3 comfyui-qwen-image:v0.38.0 -I \
    /tools/convert-flux-diffusers.py /src /dst/<name>.safetensors
"""
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/opt/ComfyUI")
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
import comfy.utils  # noqa: E402

FP8 = torch.float8_e4m3fn
DTYPE_NAMES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", FP8: "F8_E4M3"}


def main(src_dir: str, dst: str) -> None:
    src = Path(src_dir)
    index = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())
    key_to_shard = index["weight_map"]
    handles = {f: safe_open(str(src / f), "pt", device="cpu") for f in set(key_to_shard.values())}

    def get(k):
        return handles[key_to_shard[k]].get_tensor(k)

    def count(prefix):
        return len({k.split(".")[1] for k in key_to_shard if k.startswith(prefix + ".")})

    cfg = {
        "depth": count("transformer_blocks"),
        "depth_single_blocks": count("single_transformer_blocks"),
        "hidden_size": handles[key_to_shard["x_embedder.bias"]].get_slice("x_embedder.bias").get_shape()[0],
    }
    print(f"config: {cfg}", flush=True)
    sd_map = comfy.utils.flux_to_diffusers(cfg, output_prefix="")

    # output key -> [(src_key, offset|None, fun)]
    plan = defaultdict(list)
    for src_k, t in sd_map.items():
        if src_k not in key_to_shard:
            continue
        if isinstance(t, str):
            plan[t].append((src_k, None, lambda a: a))
        else:
            plan[t[0]].append((src_k, t[1], t[2] if len(t) > 2 else (lambda a: a)))
    unmapped = set(key_to_shard) - {s for v in plan.values() for s, _, _ in v}
    if unmapped:
        sys.exit(f"unmapped source keys (refusing to drop weights silently): {sorted(unmapped)[:10]}")

    def build(out_k):
        parts = plan[out_k]
        if parts[0][1] is None:
            assert len(parts) == 1, out_k
            w = parts[0][2](get(parts[0][0]))
        else:
            dim = parts[0][1][0]
            total = max(o[1] + o[2] for _, o, _ in parts)
            first = get(parts[0][0])
            shape = list(first.shape)
            shape[dim] = total
            w = torch.empty(shape, dtype=first.dtype)
            for s, o, fun in parts:
                w.narrow(o[0], o[1], o[2])[:] = fun(get(s))
        # fp8 only for the big linear weights inside the blocks; embedders,
        # norms, biases and final layer stay bf16 (same split as the usual
        # flux1-dev-fp8 releases).
        if out_k.endswith(".weight") and w.ndim == 2 and ("double_blocks." in out_k or "single_blocks." in out_k):
            return w.to(FP8)
        return w.to(torch.bfloat16)

    out_keys = sorted(plan)
    # pass 1: header (shapes/dtypes) without materialising tensors
    header, offset = {}, 0
    for k in out_keys:
        parts = plan[k]
        shp = handles[key_to_shard[parts[0][0]]].get_slice(parts[0][0]).get_shape()
        shp = list(shp)
        if parts[0][1] is not None:
            shp[parts[0][1][0]] = max(o[1] + o[2] for _, o, _ in parts)
        is_fp8 = k.endswith(".weight") and len(shp) == 2 and ("double_blocks." in k or "single_blocks." in k)
        dt = FP8 if is_fp8 else torch.bfloat16
        n = 1
        for d in shp:
            n *= d
        nbytes = n * (1 if is_fp8 else 2)
        header[k] = {"dtype": DTYPE_NAMES[dt], "shape": shp, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    header["__metadata__"] = {"format": "pt", "source": src_dir}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)

    # pass 2: stream tensors
    tmp = Path(dst + ".partial")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for i, k in enumerate(out_keys):
            w = build(k).contiguous()
            exp = header[k]
            assert list(w.shape) == exp["shape"] and DTYPE_NAMES[w.dtype] == exp["dtype"], (k, w.shape, w.dtype, exp)
            f.write(w.view(torch.uint8).numpy().tobytes())
            if i % 100 == 0:
                print(f"  {i}/{len(out_keys)} {k}", flush=True)
    tmp.rename(dst)
    print(f"wrote {dst}: {len(out_keys)} tensors, {offset / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
