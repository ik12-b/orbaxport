"""
HuggingFace Transformers-style facade for orbaxport.

Familiar patterns:
  - save_pretrained / from_pretrained-like folder layout
  - Auto-style from_orbax(...)
  - config.json + model.safetensors(+shards) + optional tokenizer copy
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .convert import convert, available_key_maps

PathLike = Union[str, Path]


@dataclass
class OrbaxPortConfig:
    """Lightweight config in the spirit of transformers.PretrainedConfig."""

    model_type: str = "orbaxport"
    architectures: List[str] = field(default_factory=lambda: ["CausalLM"])
    # conversion provenance
    source_checkpoint: Optional[str] = None
    key_map: str = "auto"
    split_qkv: bool = True
    transpose: bool = True
    torch_dtype: str = "bfloat16"
    # optional model hyperparams (user-filled)
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    intermediate_size: Optional[int] = None
    transformers_version: Optional[str] = None
    orbaxport_version: str = "1.2.0"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def save_pretrained(self, save_directory: PathLike) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrbaxPortConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in data.items() if k in valid})

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: PathLike) -> "OrbaxPortConfig":
        path = Path(pretrained_model_name_or_path)
        cfg_path = path / "config.json" if path.is_dir() else path
        data = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def save_pretrained(
    checkpoint_path: PathLike,
    save_directory: PathLike,
    *,
    # transformers-like naming
    model: Any = None,
    config: Optional[Union[OrbaxPortConfig, Dict[str, Any]]] = None,
    # conversion controls
    key_map: str = "auto",
    torch_dtype: str = "bfloat16",
    transpose: bool = True,
    split_qkv: bool = True,
    max_shard_size: str = "5GB",
    tokenizer_path: Optional[PathLike] = None,
    base_model_path: Optional[PathLike] = None,
    step: Union[str, int] = "latest",
    weights_only: bool = True,
    layers: Optional[str] = None,
    push_to_hub: bool = False,
    repo_id: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convert an Orbax/Tunix checkpoint into a HuggingFace-style folder.

    Mirrors the feel of ``model.save_pretrained(save_directory)``:

    save_directory/
      config.json
      model.safetensors  (or sharded model-00001-of-NNNNN.safetensors + index)
      tokenizer files (if tokenizer_path / base_model_path given)

    Example
    -------
    >>> from orbaxport import save_pretrained
    >>> save_pretrained(
    ...     "/path/to/tunix_ckpts/470000",
    ...     "./my-model",
    ...     key_map="gemma3",
    ...     torch_dtype="bfloat16",
    ...     tokenizer_path="google/gemma-2-2b",
    ... )
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = OrbaxPortConfig(
            source_checkpoint=str(checkpoint_path),
            key_map=key_map,
            split_qkv=split_qkv,
            transpose=transpose,
            torch_dtype=torch_dtype,
        )
    elif isinstance(config, dict):
        config = OrbaxPortConfig.from_dict(
            {
                **config,
                "source_checkpoint": str(checkpoint_path),
                "key_map": key_map,
                "torch_dtype": torch_dtype,
            }
        )
    else:
        config.source_checkpoint = str(checkpoint_path)
        config.key_map = key_map
        config.torch_dtype = torch_dtype
        config.transpose = transpose
        config.split_qkv = split_qkv

    try:
        import transformers
        config.transformers_version = getattr(transformers, "__version__", None)
    except Exception:
        pass

    base = base_model_path or tokenizer_path

    state = convert(
        input_path=checkpoint_path,
        output_path=save_directory,
        model=model,
        key_map=key_map,
        target_dtype=torch_dtype,
        split_qkv=split_qkv,
        transpose=transpose,
        max_shard_size=max_shard_size,
        hf_folder=True,
        base_model_path=base,
        step=step,
        weights_only=weights_only,
        layers=layers,
        verbose=verbose,
        **{k: v for k, v in kwargs.items() if k in (
            "item", "prefix", "lora_only", "cpu_safe", "verify", "validate", "formats",
        )},
    )

    # Prefer copying config from base HF model when available; else write ours
    cfg_out = save_directory / "config.json"
    if base and (Path(base) / "config.json").exists():
        # keep upstream config; attach orbaxport provenance alongside
        try:
            upstream = json.loads((Path(base) / "config.json").read_text(encoding="utf-8"))
        except Exception:
            upstream = {}
        upstream.setdefault("orbaxport", config.to_dict())
        cfg_out.write_text(json.dumps(upstream, indent=2) + "\n", encoding="utf-8")
        if verbose:
            print(f"[orbaxport] merged upstream config.json from {base}")
    else:
        config.save_pretrained(save_directory)

    # optional hub push
    if push_to_hub:
        if not repo_id:
            raise ValueError("push_to_hub=True requires repo_id='user/repo'")
        _push_to_hub(save_directory, repo_id=repo_id, token=token, private=private, verbose=verbose)

    return state


def from_orbax(
    checkpoint_path: PathLike,
    *,
    key_map: str = "auto",
    torch_dtype: str = "bfloat16",
    transpose: bool = True,
    model: Any = None,
    step: Union[str, int] = "latest",
    return_dict: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Load / convert an Orbax checkpoint and return a state dict (HF-style keys).

    Similar spirit to ``AutoModel.from_pretrained``, but source is Orbax/Tunix.

    Returns
    -------
    dict[str, np.ndarray]
        Weight dictionary with HuggingFace-like parameter names.
    """
    import tempfile
    out = Path(tempfile.mkdtemp(prefix="orbaxport_from_orbax_")) / "model.safetensors"
    state = convert(
        input_path=checkpoint_path,
        output_path=out,
        model=model,
        key_map=key_map,
        target_dtype=torch_dtype,
        transpose=transpose,
        step=step,
        weights_only=True,
        verbose=verbose,
        **kwargs,
    )
    return state


def load_state_dict(
    pretrained_model_name_or_path: PathLike,
    *,
    device: Optional[str] = None,
    framework: str = "numpy",
) -> Dict[str, Any]:
    """
    Load a folder or .safetensors file produced by save_pretrained / convert.

    framework: "numpy" | "torch"
    """
    path = Path(pretrained_model_name_or_path)
    from .convert import _require_safetensors

    safe_np = _require_safetensors()
    tensors: Dict[str, Any] = {}

    if path.is_file() and path.suffix == ".safetensors":
        tensors = dict(safe_np.load_file(str(path)))
    elif path.is_dir():
        index = path / "model.safetensors.index.json"
        if index.exists():
            meta = json.loads(index.read_text(encoding="utf-8"))
            weight_map = meta.get("weight_map", {})
            files = sorted(set(weight_map.values()))
            for fn in files:
                tensors.update(safe_np.load_file(str(path / fn)))
        else:
            single = path / "model.safetensors"
            if single.exists():
                tensors = dict(safe_np.load_file(str(single)))
            else:
                for f in sorted(path.glob("*.safetensors")):
                    tensors.update(safe_np.load_file(str(f)))
    else:
        raise FileNotFoundError(pretrained_model_name_or_path)

    if framework == "torch":
        import torch
        out = {k: torch.from_numpy(v) for k, v in tensors.items()}
        if device:
            out = {k: t.to(device) for k, t in out.items()}
        return out
    return tensors


def _push_to_hub(
    folder: Path,
    *,
    repo_id: str,
    token: Optional[str],
    private: bool,
    verbose: bool,
) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError(
            "push_to_hub requires huggingface_hub: pip install huggingface_hub"
        ) from e

    api = HfApi(token=token)
    create_repo(repo_id, private=private, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )
    if verbose:
        print(f"[orbaxport] pushed → https://huggingface.co/{repo_id}")


# Aliases that feel like transformers
AutoOrbaxConfig = OrbaxPortConfig
