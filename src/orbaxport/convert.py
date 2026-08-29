#!/usr/bin/env python3
"""orbaxport v1.2 – Orbax/Tunix → SafeTensors / HF (TPU-aware)."""
from __future__ import annotations
import argparse, json, os, re, shutil, sys, tempfile, time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    from .tpu_utils import (
        diff_sharding,
        estimate_checkpoint_bytes,
        is_gcs,
        print_mesh_report,
        resolve_path,
        select_step,
        tpu_slice_profile,
        detect_processes,
        read_sharding_info,
    )
except ImportError:
    from tpu_utils import (
        diff_sharding,
        estimate_checkpoint_bytes,
        is_gcs,
        print_mesh_report,
        resolve_path,
        select_step,
        tpu_slice_profile,
        detect_processes,
        read_sharding_info,
    )

def _require_jax():
    try:
        import jax, jax.numpy as jnp
        from jax.tree_util import tree_leaves_with_path, tree_map
        return jax, jnp, tree_leaves_with_path, tree_map
    except ImportError as e:
        raise ImportError("JAX required: pip install jax") from e

def _require_orbax():
    try:
        from orbax.checkpoint import v1 as ocp
        return ocp
    except ImportError:
        import orbax.checkpoint as ocp
        return ocp

def _require_safetensors():
    import safetensors.numpy as safe_np
    return safe_np

KeyTransform = Callable[[str], str]

def _compile_rules(rules):
    return [(re.compile(pat), repl) for pat, repl in rules]

def _apply_rules(key, compiled):
    for pat, repl in compiled:
        m = pat.fullmatch(key)
        if m:
            return m.expand(repl)
    return key

_GEMMA3 = [
    (r"embedder\.input_embedding(?:\.value)?", r"model.embed_tokens.weight"),
    (r"final_norm\.scale(?:\.value)?", r"model.norm.weight"),
    (r"layers\.([0-9]+)\.attn\._query_norm\.scale(?:\.value)?", r"model.layers.\1.self_attn.q_norm.weight"),
    (r"layers\.([0-9]+)\.attn\._key_norm\.scale(?:\.value)?", r"model.layers.\1.self_attn.k_norm.weight"),
    (r"layers\.([0-9]+)\.pre_attention_norm\.scale(?:\.value)?", r"model.layers.\1.input_layernorm.weight"),
    (r"layers\.([0-9]+)\.post_attention_norm\.scale(?:\.value)?", r"model.layers.\1.post_attention_layernorm.weight"),
    (r"layers\.([0-9]+)\.pre_ffw_norm\.scale(?:\.value)?", r"model.layers.\1.pre_feedforward_layernorm.weight"),
    (r"layers\.([0-9]+)\.post_ffw_norm\.scale(?:\.value)?", r"model.layers.\1.post_feedforward_layernorm.weight"),
    (r"layers\.([0-9]+)\.mlp\.gate_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.gate_proj.weight"),
    (r"layers\.([0-9]+)\.mlp\.up_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.up_proj.weight"),
    (r"layers\.([0-9]+)\.mlp\.down_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.down_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.attn_vec_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.o_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.qkv_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.qkv_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.q_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.q_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.kv_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.kv_proj.weight"),
]
_GEMMA2 = [
    (r"embedder\.input_embedding(?:\.value)?", r"model.embed_tokens.weight"),
    (r"final_norm\.scale(?:\.value)?", r"model.norm.weight"),
    (r"layers\.([0-9]+)\.pre_attention_norm\.scale(?:\.value)?", r"model.layers.\1.input_layernorm.weight"),
    (r"layers\.([0-9]+)\.post_attention_norm\.scale(?:\.value)?", r"model.layers.\1.post_attention_layernorm.weight"),
    (r"layers\.([0-9]+)\.pre_ffw_norm\.scale(?:\.value)?", r"model.layers.\1.pre_feedforward_layernorm.weight"),
    (r"layers\.([0-9]+)\.post_ffw_norm\.scale(?:\.value)?", r"model.layers.\1.post_feedforward_layernorm.weight"),
    (r"layers\.([0-9]+)\.mlp\.gate_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.gate_proj.weight"),
    (r"layers\.([0-9]+)\.mlp\.up_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.up_proj.weight"),
    (r"layers\.([0-9]+)\.mlp\.down_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.down_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.attn_vec_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.o_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.qkv_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.qkv_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.q_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.q_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.kv_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.kv_proj.weight"),
]
_LLAMA = [
    (r"embedder\.input_embedding(?:\.value)?", r"model.embed_tokens.weight"),
    (r"final_norm\.scale(?:\.value)?", r"model.norm.weight"),
    (r"lm_head\.kernel(?:\.value)?", r"lm_head.weight"),
    (r"layers\.([0-9]+)\.pre_attention_norm\.scale(?:\.value)?", r"model.layers.\1.input_layernorm.weight"),
    (r"layers\.([0-9]+)\.post_attention_norm\.scale(?:\.value)?", r"model.layers.\1.post_attention_layernorm.weight"),
    (r"layers\.([0-9]+)\.mlp\.gate_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.gate_proj.weight"),
    (r"layers\.([0-9]+)\.mlp\.up_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.up_proj.weight"),
    (r"layers\.([0-9]+)\.mlp\.down_proj\.kernel(?:\.value)?", r"model.layers.\1.mlp.down_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.q_proj\.kernel(?:\.value)?", r"model.layers.\1.self_attn.q_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.k_proj\.kernel(?:\.value)?", r"model.layers.\1.self_attn.k_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.v_proj\.kernel(?:\.value)?", r"model.layers.\1.self_attn.v_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.o_proj\.kernel(?:\.value)?", r"model.layers.\1.self_attn.o_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.attn_vec_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.o_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.q_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.q_proj.weight"),
    (r"layers\.([0-9]+)\.attn\.kv_einsum\.w(?:\.value)?", r"model.layers.\1.self_attn.kv_proj.weight"),
]
_GENERIC = [
    (r"(.*)kernel(?:\.value)?", r"\1weight"),
    (r"(.*)scale(?:\.value)?", r"\1weight"),
    (r"(.*)embedding(?:\.value)?", r"\1weight"),
    (r"(.*)\.value$", r"\1"),
]
PRESET_MAPS = {
    "gemma3": _GEMMA3, "gemma2": _GEMMA2, "gemma": _GEMMA2,
    "llama3": _LLAMA, "llama": _LLAMA, "qwen2": _LLAMA, "qwen3": _LLAMA, "qwen": _LLAMA,
    "generic": _GENERIC, "none": [],
}

def _load_keymap_file(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        import yaml
        data = yaml.safe_load(text)
    if isinstance(data, dict) and "rules" in data:
        return [(r["pattern"], r["repl"]) for r in data["rules"]]
    if isinstance(data, dict):
        return {str(k): str(v) for k, v in data.items()}
    return [(str(a), str(b)) for a, b in data]

def make_key_transform(key_map):
    if key_map is None or key_map == "none":
        return lambda k: k
    if callable(key_map) and not isinstance(key_map, type):
        return key_map
    if isinstance(key_map, str):
        p = Path(key_map)
        if p.suffix in (".yaml", ".yml", ".json") and p.exists():
            return make_key_transform(_load_keymap_file(p))
        name = key_map.lower().strip()
        if name not in PRESET_MAPS:
            raise ValueError(f"Unknown key_map '{key_map}'. Available: {sorted(PRESET_MAPS)}")
        compiled = _compile_rules(PRESET_MAPS[name])
        return lambda k: _apply_rules(k, compiled)
    if isinstance(key_map, dict):
        m = dict(key_map)
        return lambda k: m.get(k, k)
    compiled = _compile_rules(list(key_map))
    return lambda k: _apply_rules(k, compiled)

def auto_detect_key_map(keys):
    joined = " ".join(keys)
    if "_query_norm" in joined or "post_ffw_norm" in joined:
        return "gemma3"
    if "post_attention_norm" in joined and "pre_ffw_norm" in joined:
        return "gemma2"
    if "lm_head" in joined or re.search(r"layers\.\d+\.attn\.q_proj", joined):
        return "llama3"
    if "embedder.input_embedding" in joined or "layers." in joined:
        return "generic"
    return "none"

_QKV_PAT = re.compile(r"model\.layers\.([0-9]+)\.self_attn\.qkv_proj\.weight$")
_KV_PAT = re.compile(r"model\.layers\.([0-9]+)\.self_attn\.kv_proj\.weight$")
_RAW_QKV = re.compile(r"layers\.([0-9]+)\.attn\.qkv_einsum\.w(?:\.value)?$")
_RAW_KV = re.compile(r"layers\.([0-9]+)\.attn\.kv_einsum\.w(?:\.value)?$")
_LAYER_PAT = re.compile(r"(?:model\.)?layers\.([0-9]+)")
_TRANSPOSE_SUFFIXES = (
    "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.o_proj.weight", "lm_head.weight",
)

def _to_numpy(x):
    if hasattr(x, "value"):
        x = x.value
    try:
        import jax
        if isinstance(x, jax.Array):
            x = jax.device_get(x)
    except Exception:
        pass
    return np.asarray(x)

def _flatten_heads(x):
    """Flatten a per-head Q/K/V slice sliced out of a fused qkv/kv einsum kernel
    into a 2D (num_heads * head_dim, hidden_in) matrix suitable for nn.Linear.

    Slices taken from a fused qkv_einsum/kv_einsum kernel follow the Gemma/Flax
    einsum convention: shape (num_heads, hidden_in, head_dim) -- i.e. head_dim is
    always the LAST axis. Because num_heads and head_dim are not adjacent axes,
    a transpose is required before merging them; a plain .reshape() silently
    scrambles values while keeping the output shape "correct-looking".

    This intentionally no longer guesses the axis order by comparing
    shape[1] vs shape[2] (the previous heuristic), which produced wrong-but-
    same-shape tensors whenever hidden_in was numerically larger than head_dim
    (the common case).
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        num_heads, hidden_in, head_dim = x.shape
        return np.ascontiguousarray(x.transpose(0, 2, 1)).reshape(num_heads * head_dim, hidden_in)
    return x.reshape(x.shape[0], -1)

def _split_fused(state, split_qkv=True, verbose=False):
    if not split_qkv:
        return state
    out, consumed = {}, set()
    for key, arr in state.items():
        m = _QKV_PAT.fullmatch(key) or _RAW_QKV.fullmatch(key)
        if m and arr.ndim >= 1 and arr.shape[0] == 3:
            layer, prefix = m.group(1), f"model.layers.{m.group(1)}.self_attn"
            out[f"{prefix}.q_proj.weight"] = _flatten_heads(arr[0])
            out[f"{prefix}.k_proj.weight"] = _flatten_heads(arr[1])
            out[f"{prefix}.v_proj.weight"] = _flatten_heads(arr[2])
            consumed.add(key)
            if verbose: print(f"  [split] {key} → q/k/v")
            continue
        m = _KV_PAT.fullmatch(key) or _RAW_KV.fullmatch(key)
        if m and arr.ndim >= 1 and arr.shape[0] == 2:
            layer, prefix = m.group(1), f"model.layers.{m.group(1)}.self_attn"
            out[f"{prefix}.k_proj.weight"] = _flatten_heads(arr[0])
            out[f"{prefix}.v_proj.weight"] = _flatten_heads(arr[1])
            consumed.add(key)
            if verbose: print(f"  [split] {key} → k/v")
            continue
        if key not in consumed:
            out[key] = arr
    for k, v in state.items():
        if k not in consumed and k not in out:
            out[k] = v
    return out

def _maybe_transpose(state, transpose=False, verbose=False):
    if not transpose:
        return state
    out = {}
    for k, v in state.items():
        if v.ndim == 2 and any(k.endswith(s) for s in _TRANSPOSE_SUFFIXES):
            out[k] = np.ascontiguousarray(v.T)
            if verbose: print(f"  [transpose] {k}: {v.shape} → {out[k].shape}")
        else:
            out[k] = v
    return out

def _cast_array(arr, dtype):
    if not np.issubdtype(arr.dtype, np.floating):
        return arr
    t = dtype.lower()
    if t in ("bfloat16", "bf16"):
        try:
            bf = getattr(np, "bfloat16", None) or __import__("ml_dtypes").bfloat16
            return arr.astype(bf)
        except Exception:
            return arr.astype(np.float32)
    if t in ("float16", "fp16"): return arr.astype(np.float16)
    if t in ("float32", "fp32"): return arr.astype(np.float32)
    return arr.astype(np.dtype(dtype))

def _cast_dict(d, dtype):
    return d if dtype is None else {k: _cast_array(v, dtype) for k, v in d.items()}

def _filter_layers(state, layers):
    if not layers:
        return state
    allowed = set()
    for part in layers.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a, b = part.split("-", 1)
            allowed.update(range(int(a), int(b) + 1))
        else:
            allowed.add(int(part))
    out = {}
    for k, v in state.items():
        m = _LAYER_PAT.search(k)
        if m is None or int(m.group(1)) in allowed:
            out[k] = v
    return out

def _filter_lora_only(state):
    return {k: v for k, v in state.items() if "lora" in k.lower()}

def _is_mostly_lora(keys):
    keys = list(keys)
    return bool(keys) and sum(1 for k in keys if "lora" in k.lower()) / len(keys) > 0.5

def _safe_save_file(state, path, metadata=None):
    safe_np = _require_safetensors()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".safetensors", dir="/tmp")
    os.close(fd)
    tmp = Path(tmp)
    try:
        meta = {k: str(v) for k, v in (metadata or {}).items()} or None
        safe_np.save_file(state, str(tmp), metadata=meta)
        shutil.move(str(tmp), str(path))
    finally:
        if tmp.exists():
            try: tmp.unlink()
            except OSError: pass


SUPPORTED_FORMATS = ("safetensors", "pytorch", "numpy", "msgpack", "pickle", "pack")

def _to_float_safe(arr: np.ndarray) -> np.ndarray:
    """Cast bf16 to float32 for formats that do not support bfloat16."""
    if str(arr.dtype) in ("bfloat16", "bf16") or getattr(arr.dtype, "name", "") == "bfloat16":
        return arr.astype(np.float32)
    return arr

def _save_pytorch(state, path: Path, verbose=False):
    import torch
    path = Path(path)
    if path.suffix not in (".pt", ".pth", ".bin"):
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = {}
    for k, v in state.items():
        a = _to_float_safe(np.ascontiguousarray(v))
        sd[k] = torch.from_numpy(a)
    # atomic-ish via tmp
    fd, tmp = tempfile.mkstemp(suffix=".pt", dir="/tmp")
    os.close(fd)
    try:
        torch.save(sd, tmp)
        shutil.move(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass
    if verbose:
        print(f"  [pytorch] {path}")
    return path

def _save_numpy(state, path: Path, verbose=False):
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: _to_float_safe(np.ascontiguousarray(v)) for k, v in state.items()}
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir="/tmp")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **payload)
        # np.savez adds .npz if missing; handle both
        produced = tmp if Path(tmp).exists() else tmp + ".npz"
        if not Path(produced).exists() and Path(tmp + ".npz").exists():
            produced = tmp + ".npz"
        shutil.move(str(produced), str(path))
    finally:
        for p in (tmp, tmp + ".npz"):
            if os.path.exists(p):
                try: os.unlink(p)
                except OSError: pass
    if verbose:
        print(f"  [numpy] {path}")
    return path

def _save_msgpack(state, path: Path, verbose=False):
    """Flax-compatible-ish msgpack of numpy arrays (custom ext)."""
    import msgpack
    path = Path(path)
    if path.suffix not in (".msgpack", ".mpk", ".mcpack"):
        path = path.with_suffix(".msgpack")
    path.parent.mkdir(parents=True, exist_ok=True)

    def encode(obj):
        if isinstance(obj, np.ndarray):
            a = _to_float_safe(np.ascontiguousarray(obj))
            return {
                b"__ndarray__": True,
                b"dtype": a.dtype.str,
                b"shape": list(a.shape),
                b"data": a.tobytes(),
            }
        return obj

    packed = {k: encode(v) for k, v in state.items()}
    fd, tmp = tempfile.mkstemp(suffix=".msgpack", dir="/tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            msgpack.pack(packed, f, use_bin_type=True)
        shutil.move(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass
    if verbose:
        print(f"  [msgpack] {path}")
    return path

def _save_pickle(state, path: Path, verbose=False):
    import pickle
    path = Path(path)
    if path.suffix not in (".pkl", ".pickle"):
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: _to_float_safe(np.ascontiguousarray(v)) for k, v in state.items()}
    fd, tmp = tempfile.mkstemp(suffix=".pkl", dir="/tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        shutil.move(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try: os.unlink(tmp)
            except OSError: pass
    if verbose:
        print(f"  [pickle] {path}")
    return path

def _save_pack(state, path: Path, formats, metadata=None, verbose=False):
    """Zip archive containing multiple weight formats (mcpack-style bundle)."""
    import zipfile
    path = Path(path)
    if path.suffix not in (".zip", ".pack", ".mcpack"):
        path = path.with_suffix(".mcpack")
    path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="o2s_pack_", dir="/tmp"))
    try:
        written = []
        for fmt in formats:
            if fmt in ("pack", "mcpack"):
                continue
            if fmt == "safetensors":
                p = work / "model.safetensors"
                _safe_save_file(state, p, metadata)
            elif fmt == "pytorch":
                p = _save_pytorch(state, work / "model.pt")
            elif fmt == "numpy":
                p = _save_numpy(state, work / "model.npz")
            elif fmt == "msgpack":
                p = _save_msgpack(state, work / "model.msgpack")
            elif fmt == "pickle":
                p = _save_pickle(state, work / "model.pkl")
            else:
                continue
            written.append(Path(p).name)
        # manifest
        manifest = {
            "format": "orbaxport-pack",
            "version": "1.0",
            "contents": written,
            "metadata": {k: str(v) for k, v in (metadata or {}).items()},
            "n_tensors": len(state),
            "keys_sample": list(state.keys())[:20],
        }
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        fd, tmp = tempfile.mkstemp(suffix=".mcpack", dir="/tmp")
        os.close(fd)
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in work.iterdir():
                zf.write(f, arcname=f.name)
        shutil.move(tmp, str(path))
        if verbose:
            print(f"  [pack/mcpack] {path} contents={written}")
        return path
    finally:
        shutil.rmtree(work, ignore_errors=True)

def _normalize_formats(fmt) -> list:
    if fmt is None:
        return ["safetensors"]
    if isinstance(fmt, (list, tuple)):
        parts = list(fmt)
    else:
        parts = [x.strip().lower() for x in str(fmt).split(",") if x.strip()]
    # aliases
    out = []
    for p in parts:
        if p in ("mcpack", "bundle", "zip"):
            out.append("pack")
        elif p in ("pt", "pth", "torch", "pytorch"):
            out.append("pytorch")
        elif p in ("npz", "np", "numpy"):
            out.append("numpy")
        elif p in ("mpk", "msgpack", "messagepack"):
            out.append("msgpack")
        elif p in ("pkl", "pickle"):
            out.append("pickle")
        elif p in ("st", "safetensor", "safetensors"):
            out.append("safetensors")
        else:
            out.append(p)
    # unique preserve order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq or ["safetensors"]

def _save_state_multi(state, output_path: Path, formats, metadata=None, verbose=False):
    """Write state in one or more formats. Returns list of paths written."""
    formats = _normalize_formats(formats)
    output_path = Path(output_path)
    written = []

    # If pack requested, build archive of the *other* formats
    if "pack" in formats:
        others = [f for f in formats if f != "pack"] or ["safetensors", "pytorch", "numpy"]
        p = _save_pack(state, output_path, others, metadata=metadata, verbose=verbose)
        written.append(p)
        return written

    multi = len(formats) > 1
    base = output_path
    # if user gave a file with extension, stem dir for multi
    for fmt in formats:
        if multi:
            if base.suffix:
                out = base.parent / f"{base.stem}.{_ext_for(fmt)}"
            else:
                base.mkdir(parents=True, exist_ok=True)
                out = base / f"model.{_ext_for(fmt)}"
        else:
            out = base
            if out.suffix == "" and fmt != "safetensors":
                out = out.with_suffix("." + _ext_for(fmt))
            elif out.suffix == "" and fmt == "safetensors":
                out = out.with_suffix(".safetensors")

        if fmt == "safetensors":
            if out.suffix != ".safetensors":
                out = out.with_suffix(".safetensors")
            _safe_save_file(state, out, metadata)
            written.append(out)
            if verbose: print(f"  [safetensors] {out}")
        elif fmt == "pytorch":
            written.append(_save_pytorch(state, out, verbose=verbose))
        elif fmt == "numpy":
            written.append(_save_numpy(state, out, verbose=verbose))
        elif fmt == "msgpack":
            written.append(_save_msgpack(state, out, verbose=verbose))
        elif fmt == "pickle":
            written.append(_save_pickle(state, out, verbose=verbose))
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Choose from {SUPPORTED_FORMATS}")
    return written

def _ext_for(fmt: str) -> str:
    return {
        "safetensors": "safetensors",
        "pytorch": "pt",
        "numpy": "npz",
        "msgpack": "msgpack",
        "pickle": "pkl",
        "pack": "mcpack",
    }.get(fmt, fmt)


def _parse_size(s):
    s = s.strip().upper()
    for u, m in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1)):
        if s.endswith(u):
            return int(float(s[:-len(u)]) * m)
    return int(s)

def _shard_state_dict(state, max_shard_size):
    items = sorted(state.items(), key=lambda kv: -kv[1].nbytes)
    shards, sizes, weight_map = [], [], {}
    for name, arr in items:
        n = int(arr.nbytes)
        placed = False
        for i, sz in enumerate(sizes):
            if sz + n <= max_shard_size or not shards[i]:
                shards[i][name] = arr
                sizes[i] += n
                placed = True
                break
        if not placed:
            shards.append({name: arr})
            sizes.append(n)
    n = len(shards)
    for i, shard in enumerate(shards):
        fname = f"model-{i+1:05d}-of-{n:05d}.safetensors" if n > 1 else "model.safetensors"
        for k in shard:
            weight_map[k] = fname
    return shards, weight_map

def _is_tunix_step_dir(path):
    return (path / "model_params").is_dir() or (path / "optimizer_state").is_dir()

def _is_orbax_checkpoint(path):
    markers = ("_CHECKPOINT_METADATA", "manifest.ocdbt", "_sharding", "ocdbt.process_0", "orbax.checkpoint")
    if any((path / m).exists() for m in markers):
        return True
    try:
        return any(c.is_dir() and c.name.isdigit() for c in path.iterdir())
    except Exception:
        return False

def _find_latest_step(root):
    steps = [(int(p.name), p) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    if not steps:
        raise FileNotFoundError(f"No step folders under {root}")
    return sorted(steps)[-1][1]

def _resolve_step_dir(input_path, verbose=False):
    if _is_tunix_step_dir(input_path) or (input_path / "_CHECKPOINT_METADATA").exists():
        return input_path
    if _is_orbax_checkpoint(input_path):
        try:
            step = _find_latest_step(input_path)
            if verbose: print(f"[orbaxport] Auto-selected step: {step.name}")
            return step
        except FileNotFoundError:
            return input_path
    return input_path

def _make_replicated_sharding(jax):
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ("x",))
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

def _to_abstract(tree, sharding, jax, tree_map):
    def c(x):
        if hasattr(x, "shape") and hasattr(x, "dtype"):
            return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding)
        if hasattr(x, "value") and hasattr(x.value, "shape"):
            return jax.ShapeDtypeStruct(x.value.shape, x.value.dtype, sharding=sharding)
        return x
    return tree_map(c, tree)

def _flatten_to_dict(tree, tree_leaves_with_path, prefix="", key_transform=None):
    flat = {}
    for path, value in tree_leaves_with_path(tree):
        parts = []
        for p in path:
            if hasattr(p, "key"): parts.append(str(p.key))
            elif hasattr(p, "name"): parts.append(str(p.name))
            elif hasattr(p, "idx"): parts.append(str(p.idx))
            else: parts.append(str(p))
        key = ".".join(parts)
        if prefix: key = f"{prefix}.{key}" if key else prefix
        if key_transform: key = key_transform(key)
        flat[key] = _to_numpy(value)
    return flat

def list_items(input_path):
    path = Path(input_path).expanduser().resolve()
    if not path.is_dir(): return []
    items = []
    for child in path.iterdir():
        if child.is_dir() and not child.name.startswith(("_", ".")):
            if any((child / m).exists() for m in ("_METADATA", "manifest.ocdbt", "array_metadatas", "d")):
                items.append(child.name)
    return sorted(items)

def list_steps(input_path):
    path = Path(input_path).expanduser().resolve()
    return sorted(int(c.name) for c in path.iterdir() if c.is_dir() and c.name.isdigit())

def available_key_maps():
    return sorted(PRESET_MAPS.keys())

def _restore_params(step_dir, item, abstract_tree, verbose):
    ocp = _require_orbax()
    use = (step_dir / item).is_dir()
    if verbose:
        print(f"[orbaxport] Layout: {'Tunix' if use else 'generic'} (item={item})")
    try:
        if use:
            if abstract_tree is not None:
                return ocp.load_checkpointables(step_dir, {item: abstract_tree})[item]
            restored = ocp.load_checkpointables(step_dir)
            if item not in restored:
                raise KeyError(f"Item '{item}' not in {list(restored.keys())}")
            return restored[item]
        return ocp.load(step_dir, abstract_tree) if abstract_tree is not None else ocp.load(step_dir)
    except Exception as e1:
        sub = step_dir / item
        if sub.is_dir():
            return ocp.load(sub, abstract_tree) if abstract_tree is not None else ocp.load(sub)
        raise RuntimeError(f"Restore failed: {e1}") from e1

def compare(path_a, path_b, rtol=1e-3, atol=1e-5, verbose=True):
    safe_np = _require_safetensors()
    path_a, path_b = Path(path_a), Path(path_b)
    def load_any(p):
        if p.suffix == ".safetensors" or (p.is_file() and "safetensor" in p.name):
            return dict(safe_np.load_file(str(p)))
        if p.is_dir() and list(p.glob("*.safetensors")):
            m = {}
            for f in sorted(p.glob("*.safetensors")):
                m.update(safe_np.load_file(str(f)))
            return m
        params = _restore_params(_resolve_step_dir(p), "model_params", None, False)
        _, _, tlwp, _ = _require_jax()
        return _flatten_to_dict(params, tlwp)
    a, b = load_any(path_a), load_any(path_b)
    ka, kb = set(a), set(b)
    only_a, only_b = sorted(ka - kb), sorted(kb - ka)
    common = sorted(ka & kb)
    shape_m, value_m, max_abs = [], [], {}
    for k in common:
        if a[k].shape != b[k].shape:
            shape_m.append((k, a[k].shape, b[k].shape)); continue
        if np.issubdtype(a[k].dtype, np.floating) or np.issubdtype(b[k].dtype, np.floating):
            x, y = a[k].astype(np.float32), b[k].astype(np.float32)
            diff = float(np.max(np.abs(x - y)))
            max_abs[k] = diff
            if not np.allclose(x, y, rtol=rtol, atol=atol):
                value_m.append(k)
        elif not np.array_equal(a[k], b[k]):
            value_m.append(k)
    report = {"only_in_a": only_a, "only_in_b": only_b, "shape_mismatch": shape_m,
              "value_mismatch": value_m, "max_abs_diff": max_abs, "n_common": len(common),
              "ok": not only_a and not only_b and not shape_m and not value_m}
    if verbose:
        print(f"Common: {report['n_common']} | only A: {len(only_a)} | only B: {len(only_b)}")
        print(f"Shape mismatch: {len(shape_m)} | Value mismatch: {len(value_m)} | OK={report['ok']}")
    return report

def _copy_hf_assets(src_dir, out_dir, verbose):
    if not src_dir or not Path(src_dir).exists(): return
    src = Path(src_dir)
    for pat in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "tokenizer.model", "vocab.json", "merges.txt", "added_tokens.json"]:
        for f in src.glob(pat):
            if f.is_file():
                shutil.copy2(f, out_dir / f.name)
                if verbose: print(f"  [copy] {f.name}")

def _write_adapter_config(out_dir, rank=16):
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
           "r": rank, "lora_alpha": rank, "lora_dropout": 0.0,
           "target_modules": ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]}
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

def convert(
    input_path, output_path, *, item="model_params", model=None, abstract=None,
    target_dtype=None, prefix="", key_map="auto", split_qkv=True, transpose=False,
    layers=None, lora_only=False, max_shard_size=None, hf_folder=False,
    base_model_path=None, formats="safetensors", overwrite=True, verify=False,
    dry_run=False, list_keys_only=False, verbose=True,
    cpu_safe=True, weights_only=False, step="latest",
    mesh_report=False, save_cpu_orbax=None, validate=False,
):
    jax, jnp, tree_leaves_with_path, tree_map = _require_jax()
    t0 = time.time()
    # GCS-aware path resolve
    if is_gcs(input_path):
        input_path = resolve_path(input_path)
        input_path_p = Path(str(input_path))
    else:
        input_path_p = Path(input_path).expanduser().resolve()
        if not input_path_p.exists():
            raise FileNotFoundError(input_path_p)

    output_path = Path(output_path).expanduser().resolve()

    # weights_only forces model_params
    if weights_only:
        item = "model_params"

    # Smart step selection
    try:
        step_dir = select_step(input_path_p, step=step)
    except Exception:
        step_dir = _resolve_step_dir(input_path_p, verbose=verbose)
    step_dir = Path(str(step_dir))
    if verbose:
        print(f"[orbaxport] Using: {step_dir}  (step selector={step!r})")

    # Process topology
    try:
        proc = detect_processes(step_dir)
        if verbose:
            print(f"[orbaxport] processes={proc.get('n_processes')} multi={proc.get('multi_process')}")
    except Exception:
        proc = {}

    # RAM estimate
    try:
        est = estimate_checkpoint_bytes(step_dir)
        if verbose:
            print(f"[orbaxport] size≈{est['human_estimate']} disk={est['human_disk']} RAM free={est['human_ram_free']} fits={est['fits_in_ram']}")
        if not est.get("fits_in_ram", True):
            print("[orbaxport] WARNING: may OOM — prefer --layers / --max-shard-size / higher-RAM host")
    except Exception:
        est = {}

    if mesh_report:
        print_mesh_report(step_dir, item=item)

    replicated = _make_replicated_sharding(jax)
    abstract_tree = abstract
    if abstract_tree is None and model is not None:
        if verbose: print("[orbaxport] Building abstract from model...")
        try:
            from flax import nnx
            concrete = nnx.state(model) if isinstance(model, nnx.Module) else model
        except Exception:
            concrete = model
        abstract_tree = _to_abstract(concrete, replicated, jax, tree_map)
    elif cpu_safe and abstract_tree is None and verbose:
        print("[orbaxport] cpu_safe=True: restore will reshard using checkpoint sharding file → host devices")

    if verbose: print("[orbaxport] Restoring...")
    params = _restore_params(step_dir, item, abstract_tree, verbose)
    raw_flat = _flatten_to_dict(params, tree_leaves_with_path, prefix=prefix)

    if key_map in (None, "auto"):
        key_map = auto_detect_key_map(raw_flat.keys())
        if verbose: print(f"[orbaxport] Auto key_map → '{key_map}'")
    transform_fn = make_key_transform(key_map)
    if verbose:
        print(f"[orbaxport] key_map={key_map} split_qkv={split_qkv} transpose={transpose} layers={layers}")

    state = {transform_fn(k): v for k, v in raw_flat.items()}
    state = _split_fused(state, split_qkv=split_qkv, verbose=verbose)
    state = _maybe_transpose(state, transpose=transpose, verbose=verbose)
    state = _filter_layers(state, layers)
    if lora_only or _is_mostly_lora(state.keys()):
        if not lora_only and verbose:
            print("[orbaxport] LoRA-heavy → filtering")
        state = _filter_lora_only(state)
        lora_only = True
    if target_dtype:
        if verbose: print(f"[orbaxport] Cast → {target_dtype}")
        state = _cast_dict(state, target_dtype)
    state = {k: np.ascontiguousarray(v) for k, v in state.items()}

    if verbose:
        total = sum(v.nbytes for v in state.values())
        print(f"[orbaxport] {len(state)} tensors, {total/1e9:.3f} GB ({time.time()-t0:.1f}s)")

    if list_keys_only or dry_run:
        for k in sorted(state):
            print(f"  {k}: {state[k].shape} {state[k].dtype}")
        return state

    meta = {"format": "orbaxport", "version": "1.0", "source": str(step_dir),
            "key_map": str(key_map), "split_qkv": str(split_qkv), "transpose": str(transpose),
            "target_dtype": str(target_dtype), "lora_only": str(lora_only)}

    if hf_folder or lora_only:
        out_dir = output_path if output_path.suffix != ".safetensors" else output_path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        if lora_only:
            _safe_save_file(state, out_dir / "adapter_model.safetensors", meta)
            _write_adapter_config(out_dir)
            if verbose: print(f"[orbaxport] LoRA adapter → {out_dir}")
        else:
            if max_shard_size:
                shards, wmap = _shard_state_dict(state, _parse_size(max_shard_size))
                n = len(shards)
                for i, shard in enumerate(shards):
                    fn = f"model-{i+1:05d}-of-{n:05d}.safetensors" if n > 1 else "model.safetensors"
                    _safe_save_file(shard, out_dir / fn, meta if i == 0 else None)
                if n > 1:
                    idx = {"metadata": {"total_size": sum(v.nbytes for v in state.values())}, "weight_map": wmap}
                    (out_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
            else:
                _safe_save_file(state, out_dir / "model.safetensors", meta)
            _copy_hf_assets(base_model_path, out_dir, verbose)
            if verbose: print(f"[orbaxport] HF folder → {out_dir}")
        if verbose: print("[orbaxport] Done.")
        return state

    # Optional: save CPU-replicated Orbax intermediate
    if save_cpu_orbax:
        try:
            ocp = _require_orbax()
            out_cpu = Path(save_cpu_orbax)
            if out_cpu.exists():
                shutil.rmtree(out_cpu)
            out_cpu.parent.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"[orbaxport] Saving CPU-replicated Orbax → {out_cpu}")
            # re-tree from flat is hard; save flat numpy via orbax if possible
            # Fallback: write a small marker + safetensors alongside
            ocp.save_checkpointables(out_cpu, {"model_params": params})
        except Exception as e:
            if verbose:
                print(f"[orbaxport] save_cpu_orbax failed ({e}); continuing export")

    fmt_list = _normalize_formats(formats)
    if verbose:
        print(f"[orbaxport] formats={fmt_list}")

    # Sharded safetensors only when single format is safetensors
    if max_shard_size and fmt_list == ["safetensors"]:
        out_dir = output_path.parent / output_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        shards, wmap = _shard_state_dict(state, _parse_size(max_shard_size))
        n = len(shards)
        for i, shard in enumerate(shards):
            fn = f"model-{i+1:05d}-of-{n:05d}.safetensors" if n > 1 else "model.safetensors"
            _safe_save_file(shard, out_dir / fn, meta if i == 0 else None)
        if n > 1:
            idx = {"metadata": {"total_size": sum(v.nbytes for v in state.values())}, "weight_map": wmap}
            (out_dir / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2))
        if verbose: print(f"[orbaxport] Sharded ({n}) → {out_dir}")
    else:
        if output_path.exists() and output_path.is_file() and not overwrite and len(fmt_list) == 1:
            raise FileExistsError(output_path)
        if verbose: print(f"[orbaxport] Writing formats → {output_path}")
        written = _save_state_multi(state, output_path, fmt_list, metadata=meta, verbose=verbose)
        if verify and "safetensors" in fmt_list:
            st_path = next((p for p in written if str(p).endswith(".safetensors")), None)
            if st_path:
                reloaded = _require_safetensors().load_file(str(st_path))
                assert set(reloaded) == set(state)
                if verbose: print("[orbaxport] Verify OK.")
    if validate:
        try:
            import jax
            # smoke: device_put first few tensors on host
            n = 0
            for k, v in list(state.items())[:5]:
                jax.device_put(jax.numpy.asarray(v.astype(np.float32) if v.dtype == np.dtype("O") else v))
                n += 1
            if verbose:
                print(f"[orbaxport] validate: device_put OK on {n} sample tensors")
        except Exception as e:
            print(f"[orbaxport] validate warning: {e}")

    if verbose: print("[orbaxport] Done.")
    return state

def main(argv=None):
    p = argparse.ArgumentParser(description="orbaxport v1.0")
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--item", default="model_params")
    p.add_argument("--dtype", default=None, choices=[None, "float32", "float16", "bfloat16"])
    p.add_argument("--prefix", default="")
    p.add_argument("--key-map", default="auto")
    p.add_argument("--no-split-qkv", action="store_true")
    p.add_argument("--transpose", action="store_true")
    p.add_argument("--layers", default=None)
    p.add_argument("--lora-only", action="store_true")
    p.add_argument("--max-shard-size", default=None)
    p.add_argument("--hf-folder", action="store_true")
    p.add_argument("--base-model-path", default=None)
    p.add_argument("--format", default="safetensors",
                   help="Output format(s), comma-separated: safetensors,pytorch,numpy,msgpack,pickle,pack/mcpack")
    p.add_argument("--cpu-safe", action="store_true", default=True,
                   help="Prefer safe CPU resharding (default on)")
    p.add_argument("--no-cpu-safe", action="store_true", help="Disable cpu-safe helpers")
    p.add_argument("--weights-only", action="store_true", help="Only model_params (skip optimizer)")
    p.add_argument("--step", default="latest", help="latest | best | <int>")
    p.add_argument("--mesh-report", action="store_true", help="Print mesh/topology/RAM report")
    p.add_argument("--save-cpu-orbax", default=None, help="Also save CPU-replicated Orbax dir")
    p.add_argument("--validate", action="store_true", help="Smoke-test device_put after export")
    p.add_argument("--diff-sharding", nargs=2, metavar=("A", "B"), help="Compare sharding of two steps")
    p.add_argument("--tpu-profile", action="store_true", help="Print TPU slice/env profile only")
    p.add_argument("--estimate", action="store_true", help="Estimate checkpoint size / RAM only")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-keys", action="store_true")
    p.add_argument("--list-steps", action="store_true")
    p.add_argument("--list-items", action="store_true")
    p.add_argument("--list-key-maps", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    p.add_argument("--no-overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    ip = Path(args.input)
    if args.list_key_maps:
        print("\n".join(available_key_maps())); return 0
    if args.list_steps:
        print(list_steps(ip)); return 0
    if args.list_items:
        print(list_items(_resolve_step_dir(ip))); return 0
    if args.compare:
        compare(args.compare[0], args.compare[1], verbose=not args.quiet); return 0
    if args.tpu_profile:
        import json as _json
        print(_json.dumps(tpu_slice_profile(), indent=2, default=str)); return 0
    if args.diff_sharding:
        import json as _json
        print(_json.dumps(diff_sharding(args.diff_sharding[0], args.diff_sharding[1], item=args.item), indent=2, default=str)); return 0
    if args.estimate or args.mesh_report:
        step_dir = select_step(args.input, step=args.step)
        if args.mesh_report or args.estimate:
            print_mesh_report(step_dir, item=args.item)
        if args.estimate and not args.output:
            return 0
        if args.mesh_report and not args.output:
            return 0
    if args.output is None and not (args.dry_run or args.list_keys):
        p.error("--output required")
    convert(
        input_path=args.input, output_path=args.output or "/tmp/_dry.safetensors",
        item=args.item, target_dtype=args.dtype, prefix=args.prefix, key_map=args.key_map,
        split_qkv=not args.no_split_qkv, transpose=args.transpose, layers=args.layers,
        lora_only=args.lora_only, max_shard_size=args.max_shard_size, hf_folder=args.hf_folder,
        base_model_path=args.base_model_path, formats=args.format,
        overwrite=not args.no_overwrite, verify=args.verify,
        dry_run=args.dry_run, list_keys_only=args.list_keys, verbose=not args.quiet,
        cpu_safe=not args.no_cpu_safe, weights_only=args.weights_only, step=args.step,
        mesh_report=args.mesh_report, save_cpu_orbax=args.save_cpu_orbax, validate=args.validate,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
