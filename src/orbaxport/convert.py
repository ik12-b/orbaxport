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
# Suffixes of standalone (non-fused) 3D attention projections that _split_fused
# never touches because they don't match the fused qkv/kv patterns. These still
# need head-flattening -- see _flatten_remaining_3d / Issue #2.
_QKV_ROLE_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
)
_O_PROJ_SUFFIX = "self_attn.o_proj.weight"

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

def _resolve_head_axis(x, role, head_dim=None, key="", verbose=False):
    """Determine which axis of a 3D per-head tensor is head_dim, instead of
    blindly trusting a fixed position.

    role: "qkv" (input-style projections: q/k/v) or "o" (output projection).
    The two roles use OPPOSITE axis conventions in the Gemma/Flax einsum
    layout this tool targets by default -- head_dim is the LAST axis for
    q/k/v (shape (heads, hidden_in, head_dim)) but the MIDDLE axis for o
    (shape (heads, head_dim, hidden_out)) -- but that convention is a
    property of one model family's einsum layout, not a law of nature.
    Other JAX/Flax attention implementations may lay these axes out
    differently.

    If `head_dim` is given (explicitly, or inferred elsewhere in this layer
    and passed through), this picks whichever of the two candidate axes
    (index 1 or 2) actually has that size, so the correct axis is found by
    matching real dimensions rather than assumed position -- this is what
    makes the tool generalize beyond the one convention it was originally
    written against. If neither axis matches (or head_dim is None), it
    falls back to the documented positional convention for the given role,
    optionally warning that it's an assumption rather than a verified fact.

    Returns the resolved head_dim axis index (1 or 2).
    """
    default_axis = 2 if role == "qkv" else 1
    if head_dim is None:
        return default_axis
    _, a, b = x.shape
    matches = [i for i, sz in ((1, a), (2, b)) if sz == head_dim]
    if len(matches) == 1:
        axis = matches[0]
        if axis != default_axis and verbose:
            print(f"  [head-dim] {key}: shape {x.shape} matches head_dim={head_dim} "
                  f"on axis {axis}, not the default {role}-role axis {default_axis} "
                  f"-- using the matched axis")
        return axis
    if len(matches) == 2 and verbose:
        print(f"  [head-dim] {key}: shape {x.shape} -- both axis 1 and 2 equal "
              f"head_dim={head_dim} (ambiguous), falling back to the default "
              f"{role}-role axis {default_axis}")
    elif not matches and verbose:
        print(f"  [head-dim] {key}: shape {x.shape} -- neither axis matches the "
              f"given head_dim={head_dim}, falling back to the default {role}-role "
              f"axis {default_axis} (this checkpoint may use a different attention "
              f"layout than assumed; verify the output with --validate)")
    return default_axis

def _flatten_heads(x, head_dim=None, key="", verbose=False):
    """Flatten a per-head Q/K/V-role tensor into a 2D Flax-style
    (hidden_in, num_heads * head_dim) == (in_features, out_features) matrix.

    By default this assumes the Gemma/Flax einsum convention -- shape
    (num_heads, hidden_in, head_dim), head_dim as the LAST axis. Pass an
    explicit `head_dim` (see _resolve_head_axis) to make this robust to
    architectures that don't follow that convention.

    Output is intentionally kept in Flax [in, out] orientation, matching every
    other (non-split) kernel in the state dict at this point in the pipeline
    (e.g. mlp.gate_proj.kernel). The single, later `_maybe_transpose` pass is
    what converts everything -- split or not -- to HF [out, in] when requested.
    Returning HF-oriented output here directly (as an earlier version of this
    function did) causes q/k/v/o tensors to get transposed *twice* when
    transpose=True, silently flipping them back into the wrong orientation.

    Because num_heads and head_dim are not adjacent axes in the input, a
    transpose is required before merging them -- a plain .reshape() would
    silently scramble values while keeping the output shape "correct-looking".
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        num_heads = x.shape[0]
        head_axis = _resolve_head_axis(x, "qkv", head_dim, key, verbose)
        if head_axis == 2:
            hidden_in = x.shape[1]
            return np.ascontiguousarray(x.transpose(1, 0, 2)).reshape(hidden_in, num_heads * x.shape[2])
        hidden_in = x.shape[2]
        return np.ascontiguousarray(x.transpose(2, 0, 1)).reshape(hidden_in, num_heads * x.shape[1])
    return x.reshape(x.shape[0], -1)

def _flatten_heads_o(x, head_dim=None, key="", verbose=False):
    """Flatten the output-projection tensor into a 2D Flax-style
    (num_heads * head_dim, hidden_out) == (in_features, out_features) matrix.

    By default this assumes the Gemma/Flax einsum convention -- shape
    (num_heads, head_dim, hidden_out), head_dim as the MIDDLE axis, already
    adjacent to num_heads so a plain reshape (no transpose) suffices. Pass an
    explicit `head_dim` to make this robust to architectures where that
    axis isn't in the assumed position.
    """
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        num_heads = x.shape[0]
        head_axis = _resolve_head_axis(x, "o", head_dim, key, verbose)
        if head_axis == 1:
            hidden_out = x.shape[2]
            return np.ascontiguousarray(x).reshape(num_heads * x.shape[1], hidden_out)
        hidden_out = x.shape[1]
        return np.ascontiguousarray(x.transpose(0, 2, 1)).reshape(num_heads * x.shape[2], hidden_out)
    return x.reshape(x.shape[0], -1)

def _flatten_remaining_3d(state, head_dim=None, verbose=False):
    """Flatten any 3D attention-projection tensor that _split_fused left
    untouched -- i.e. standalone q_proj (from q_einsum) and o_proj (from
    attn_vec_einsum) on architectures that use separate Q/KV einsum kernels
    instead of one fused 3-way qkv_einsum (the standard layout for GQA models
    such as Gemma2/Gemma3, where Q and KV have different head counts).

    Without this pass these tensors reach the safetensors output as rank-3
    arrays instead of the rank-2 shape nn.Linear requires, and loading the
    checkpoint into a HF model fails outright on a shape mismatch. See Issue #2.
    """
    out = {}
    for k, v in state.items():
        if v.ndim == 3 and k.endswith(_O_PROJ_SUFFIX):
            out[k] = _flatten_heads_o(v, head_dim=head_dim, key=k, verbose=verbose)
            if verbose: print(f"  [flatten-o] {k}: {v.shape} → {out[k].shape}")
        elif v.ndim == 3 and any(k.endswith(s) for s in _QKV_ROLE_SUFFIXES):
            out[k] = _flatten_heads(v, head_dim=head_dim, key=k, verbose=verbose)
            if verbose: print(f"  [flatten-qkv] {k}: {v.shape} → {out[k].shape}")
        else:
            out[k] = v
    return out

def _infer_head_dim_from_state(state):
    """Best-effort auto-detection of head_dim, used when the caller doesn't
    pass one explicitly. Cross-checks the axis the qkv-role convention would
    pick against the axis the o-role convention would pick, for tensors in
    the same layer -- under the default Gemma/Flax convention these are
    DIFFERENT axes of DIFFERENT tensors that must agree on one head_dim
    value. If they disagree, this checkpoint likely doesn't follow the
    assumed convention; returns None so callers fall back to the plain
    positional default per-tensor (current behavior) rather than forcing a
    possibly-wrong shared guess.
    """
    by_layer = {}
    for k, v in state.items():
        if getattr(v, "ndim", None) != 3:
            continue
        m = _LAYER_PAT.search(k)
        if not m:
            continue
        layer = m.group(1)
        if k.endswith(_O_PROJ_SUFFIX):
            by_layer.setdefault(layer, {})["o"] = v.shape[1]  # default o-role axis
        elif any(k.endswith(s) for s in _QKV_ROLE_SUFFIXES):
            by_layer.setdefault(layer, {})["qkv"] = v.shape[2]  # default qkv-role axis
    candidates = set()
    for pair in by_layer.values():
        # Only trust a value that comes from genuine cross-validation --
        # TWO INDEPENDENT default-axis readings (qkv-role vs o-role) that
        # agree. A layer contributing only ONE of the two roles (e.g. q/k/v
        # is fused -- ndim==4 at this stage, invisible here -- and only
        # o_proj is standalone, which is the common case for every Gemma-
        # style layer regardless of whether it uses GQA) must NOT contribute
        # its single, unvalidated default-axis reading to `candidates`.
        #
        # Doing so was a real bug: hidden_in (q's axis 1) and hidden_out
        # (o's axis 2) are typically the SAME value (both are the model's
        # hidden_size, tied to the same residual stream) in virtually every
        # transformer architecture. So if o_proj alone didn't actually follow
        # the assumed middle-axis convention for THIS checkpoint, its
        # (wrongly-read) "head_dim" would very plausibly collide with q/k/v's
        # own hidden_in size -- causing _resolve_head_axis to flip a
        # previously-safe, correctly-defaulted q/k/v tensor onto the WRONG
        # axis, silently corrupting values on what used to be the reliable
        # default path. Only a genuine two-role agreement is real evidence.
        if "qkv" in pair and "o" in pair:
            if pair["qkv"] != pair["o"]:
                return None  # disagreement -- don't force a shared guess
            candidates.add(pair["qkv"])
    return candidates.pop() if len(candidates) == 1 else None

def _split_fused(state, split_qkv=True, head_dim=None, verbose=False):
    if not split_qkv:
        return state
    out, consumed = {}, set()
    for key, arr in state.items():
        m = _QKV_PAT.fullmatch(key) or _RAW_QKV.fullmatch(key)
        if m and arr.ndim >= 1 and arr.shape[0] == 3:
            layer, prefix = m.group(1), f"model.layers.{m.group(1)}.self_attn"
            out[f"{prefix}.q_proj.weight"] = _flatten_heads(arr[0], head_dim=head_dim, key=f"{prefix}.q_proj.weight", verbose=verbose)
            out[f"{prefix}.k_proj.weight"] = _flatten_heads(arr[1], head_dim=head_dim, key=f"{prefix}.k_proj.weight", verbose=verbose)
            out[f"{prefix}.v_proj.weight"] = _flatten_heads(arr[2], head_dim=head_dim, key=f"{prefix}.v_proj.weight", verbose=verbose)
            consumed.add(key)
            if verbose: print(f"  [split] {key} → q/k/v")
            continue
        m = _KV_PAT.fullmatch(key) or _RAW_KV.fullmatch(key)
        if m and arr.ndim >= 1 and arr.shape[0] == 2:
            layer, prefix = m.group(1), f"model.layers.{m.group(1)}.self_attn"
            out[f"{prefix}.k_proj.weight"] = _flatten_heads(arr[0], head_dim=head_dim, key=f"{prefix}.k_proj.weight", verbose=verbose)
            out[f"{prefix}.v_proj.weight"] = _flatten_heads(arr[1], head_dim=head_dim, key=f"{prefix}.v_proj.weight", verbose=verbose)
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

def _verify_safetensors_roundtrip(reloaded, state, verbose=False):
    """Real corruption check: key-set equality AND shape+value equality against
    the in-memory state that was written. The previous version of this check
    only compared key sets (`set(reloaded) == set(state)`), which catches a
    dropped/renamed key but says nothing about a truncated write, a wrong
    dtype cast, or any other value-level corruption during serialization --
    "Verify OK" could pass even if every tensor's contents were wrong.
    Raises AssertionError with a specific, actionable message on any mismatch.
    """
    missing = set(state) - set(reloaded)
    extra = set(reloaded) - set(state)
    assert not missing, f"verify: {len(missing)} key(s) missing from written output: {sorted(missing)[:5]}..."
    assert not extra, f"verify: {len(extra)} unexpected extra key(s) in written output: {sorted(extra)[:5]}..."
    shape_mismatches = [k for k in state if reloaded[k].shape != state[k].shape]
    assert not shape_mismatches, f"verify: shape mismatch on {len(shape_mismatches)} key(s): {shape_mismatches[:5]}..."
    value_mismatches = [k for k in state if not np.array_equal(reloaded[k], state[k])]
    assert not value_mismatches, f"verify: value mismatch on {len(value_mismatches)} key(s) after round-trip: {value_mismatches[:5]}..."
    if verbose:
        print(f"[orbaxport] Verify OK ({len(state)} tensors, keys+shapes+values all match).")

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

def _try_build_abstract_from_metadata(step_dir, item, sharding, jax, tree_map, verbose):
    """Best-effort: build a replicated abstract pytree straight from checkpoint
    metadata, without requiring a live model. This is what makes cpu_safe=True
    actually do something when no model/abstract is passed -- previously
    cpu_safe was read in exactly one place purely to decide whether to print a
    message, with no effect on restore behavior either way (see Issue: cpu_safe
    no-op). Returns None if the installed orbax version doesn't expose a usable
    metadata API; callers fall back to an unsharded restore in that case, same
    as the old behavior.

    Verified against a real orbax.checkpoint v1 (0.12.4) install:
    `ocp.checkpointables_metadata(step_dir).metadata` is a dict keyed by
    checkpointable name (e.g. "model_params") whose leaves are ArrayMetadata
    objects (with .shape/.dtype). It must be unwrapped through BOTH the
    CheckpointMetadata wrapper's `.metadata` attribute AND the item key --
    passing the wrapper object itself (or the whole multi-item dict) straight
    into _to_abstract silently produces a bogus "abstract tree" that is really
    just the opaque wrapper object, which orbax then rejects with an
    UnregisteredTypeError at restore time.
    """
    ocp = _require_orbax()
    candidates = []
    if hasattr(ocp, "checkpointables_metadata"):
        candidates.append(lambda: ocp.checkpointables_metadata(step_dir))
    if hasattr(ocp, "metadata"):
        candidates.append(lambda: ocp.metadata(step_dir))
    for get_meta in candidates:
        try:
            wrapper = get_meta()
            meta = getattr(wrapper, "metadata", wrapper)
        except Exception:
            continue
        if meta is None:
            continue
        if isinstance(meta, dict) and item in meta:
            meta = meta[item]
        is_leaf = lambda x: hasattr(x, "shape") and hasattr(x, "dtype")
        try:
            return tree_map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding) if is_leaf(x) else x,
                meta, is_leaf=is_leaf,
            )
        except TypeError:
            # tree_map signature without is_leaf support (older jax) -- fall
            # back to the plain _to_abstract path, which works as long as
            # `meta` doesn't contain any non-leaf objects tree_map can't see
            # into cleanly (true for the common ArrayMetadata-leaf case).
            try:
                return _to_abstract(meta, sharding, jax, tree_map)
            except Exception:
                continue
        except Exception:
            continue
    if verbose:
        print("[orbaxport] cpu_safe=True: no usable metadata API on this orbax "
              "version — falling back to default restore. Pass model=... (or "
              "abstract=...) for guaranteed CPU-safe resharding of TPU-sharded "
              "checkpoints.")
    return None

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
    head_dim=None,
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
    elif abstract_tree is None and cpu_safe:
        # No live model was given -- previously this branch only printed a
        # message and restore proceeded unsharded regardless of cpu_safe.
        # Now actually attempt to build a replicated abstract tree from the
        # checkpoint's own metadata, so TPU-sharded checkpoints can still be
        # resharded onto host devices without requiring model=...
        if verbose: print("[orbaxport] cpu_safe=True: attempting metadata-based reshard to host devices...")
        abstract_tree = _try_build_abstract_from_metadata(step_dir, item, replicated, jax, tree_map, verbose)
    elif abstract_tree is None and not cpu_safe and verbose:
        print("[orbaxport] cpu_safe=False: skipping reshard attempt, restoring with checkpoint's native sharding")

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
    resolved_head_dim = head_dim
    if resolved_head_dim is None and split_qkv:
        # Auto-infer by cross-checking the axis the q/k/v-role convention
        # would pick against the axis the o-role convention would pick, for
        # tensors in the same layer -- they must agree on one head_dim value.
        # If the checkpoint doesn't match the positional convention this tool
        # defaults to, this either finds the *actual* head_dim (generalizing
        # beyond the hardcoded Gemma/Flax layout) or safely detects
        # disagreement and backs off to the plain per-tensor default instead
        # of forcing a wrong shared guess.
        resolved_head_dim = _infer_head_dim_from_state(state)
        if verbose and resolved_head_dim is not None:
            print(f"[orbaxport] Auto-detected head_dim={resolved_head_dim} from checkpoint shapes")
    state = _split_fused(state, split_qkv=split_qkv, head_dim=resolved_head_dim, verbose=verbose)
    if split_qkv:
        # Handles standalone q_proj/o_proj tensors that arise from architectures
        # using separate q_einsum + kv_einsum kernels (GQA, e.g. Gemma2/Gemma3) --
        # _split_fused only reshapes the fused 3-way/2-way case. See Issue #2.
        state = _flatten_remaining_3d(state, head_dim=resolved_head_dim, verbose=verbose)
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
        if verify:
            # Previously this branch had NO verify logic at all -- --verify
            # combined with --max-shard-size silently checked nothing, giving
            # false confidence that the output was validated. Reload every
            # shard file and merge before checking, so a key split across
            # shards is still fully accounted for.
            safe_np = _require_safetensors()
            reloaded = {}
            for i in range(n):
                fn = f"model-{i+1:05d}-of-{n:05d}.safetensors" if n > 1 else "model.safetensors"
                reloaded.update(safe_np.load_file(str(out_dir / fn)))
            _verify_safetensors_roundtrip(reloaded, state, verbose=verbose)
    else:
        if output_path.exists() and output_path.is_file() and not overwrite and len(fmt_list) == 1:
            raise FileExistsError(output_path)
        if verbose: print(f"[orbaxport] Writing formats → {output_path}")
        written = _save_state_multi(state, output_path, fmt_list, metadata=meta, verbose=verbose)
        if verify and "safetensors" in fmt_list:
            st_path = next((p for p in written if str(p).endswith(".safetensors")), None)
            if st_path:
                reloaded = _require_safetensors().load_file(str(st_path))
                _verify_safetensors_roundtrip(reloaded, state, verbose=verbose)
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
    p.add_argument("--head-dim", type=int, default=None,
                    help="Explicit head_dim for splitting fused Q/K/V/O per-head "
                         "tensors. By default the tool assumes the Gemma/Flax "
                         "einsum axis convention and auto-cross-checks it across "
                         "q/k/v vs o tensors in the same layer; pass this to "
                         "override for checkpoints that use a different layout.")
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
        if args.estimate:
            print_mesh_report(step_dir, item=args.item)
        elif args.mesh_report and not args.output:
            # Only pre-print here when we're about to return early without
            # calling convert() below. When --output is also given, convert()
            # prints the same report itself (mesh_report=args.mesh_report is
            # forwarded to it) -- printing it here too would just duplicate it.
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
        head_dim=args.head_dim,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
