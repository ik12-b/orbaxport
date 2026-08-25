# orbaxport

[![PyPI](https://img.shields.io/pypi/v/orbaxport.svg)](https://pypi.org/project/orbaxport/)
[![Python](https://img.shields.io/pypi/pyversions/orbaxport.svg)](https://pypi.org/project/orbaxport/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Convert **Orbax / Tunix (JAX)** training checkpoints — including those saved with **TPU multi-sharding** — into:

| Format | Extension | Notes |
|--------|-----------|--------|
| SafeTensors | `.safetensors` | Default, HF-compatible |
| PyTorch | `.pt` | `torch` state_dict |
| NumPy | `.npz` | Compressed archive |
| MsgPack | `.msgpack` | Portable binary |
| Pickle | `.pkl` | Python-only |
| **mcpack** | `.mcpack` | Zip bundle of several formats |
| HF folder | directory | Weights + optional tokenizer/config |
| Sharded ST | `model-00001-of-N` | Large models |

Built for the common pain point: **checkpoint written on TPU (e.g. 8-way sharding) → convert/load on CPU**.

## Install

```bash
pip install orbaxport

# optional extras
pip install "orbaxport[torch]"      # PyTorch export
pip install "orbaxport[msgpack]"    # msgpack export
pip install "orbaxport[all]"        # torch + msgpack + yaml + flax
```

From source:

```bash
git clone https://github.com/YOUR_USERNAME/orbaxport.git
cd orbaxport
pip install -e ".[all,dev]"
```

## Transformers-style API

```python
from orbaxport import save_pretrained, from_orbax, load_state_dict, OrbaxPortConfig

# Like model.save_pretrained(...)
save_pretrained(
    "/path/to/tunix_ckpts/470000",
    "./my-hf-model",
    key_map="gemma3",
    torch_dtype="bfloat16",
    max_shard_size="5GB",
    tokenizer_path="/path/to/original_hf_model",  # copies tokenizer + merges config
)

# Load weights back (numpy or torch)
state = load_state_dict("./my-hf-model", framework="torch")

# One-shot convert → HF key dict (no folder)
state = from_orbax("/path/to/tunix_ckpts/470000", key_map="auto")
```

Folder layout (HuggingFace-compatible):

```
my-hf-model/
  config.json
  model.safetensors          # or model-00001-of-0000N.safetensors + index.json
  tokenizer.json             # if tokenizer_path / base_model_path given
  tokenizer_config.json
  ...
```

## Quick start

### CLI

```bash
# Inspect checkpoint
orbaxport -i /path/to/tunix_ckpts --list-steps
orbaxport -i /path/to/tunix_ckpts/470000 --list-items
orbaxport -i /path/to/tunix_ckpts/470000 --list-keys --dry-run

# Convert (auto key-map, TPU→CPU friendly)
orbaxport \
  -i /path/to/tunix_ckpts/470000 \
  -o model.safetensors \
  --key-map auto \
  --dtype bfloat16 \
  --verify

# HuggingFace-style folder + sharding
orbaxport \
  -i /path/to/tunix_ckpts/470000 \
  -o ./hf_model \
  --hf-folder \
  --max-shard-size 4GB \
  --base-model-path /path/to/original_hf_model \
  --transpose

# Multiple formats + mcpack bundle
orbaxport -i ./470000 -o export --format safetensors,pytorch,numpy
orbaxport -i ./470000 -o model.mcpack --format mcpack
```

### Python

```python
from orbaxport import convert, compare

# Recommended: pass your live Flax NNX model so sharding is forced to CPU
state = convert(
    input_path="/path/to/tunix_ckpts/470000",
    output_path="model.safetensors",
    model=model,                 # optional but strongly recommended
    key_map="auto",              # or "gemma3", "llama3", "qwen", ...
    split_qkv=True,
    transpose=True,              # Flax [in,out] → HF [out,in]
    target_dtype="bfloat16",
    verify=True,
)

# Partial layers (lower RAM)
convert(..., layers="0-5,10")

# LoRA adapter export
convert(..., lora_only=True, output_path="./lora_adapter")

# Compare two artifacts
report = compare("a.safetensors", "b.safetensors")
print(report["ok"])
```

## Tunix checkpoint layout

```
tunix_ckpts/
├── 460000/
├── 470000/                 ← pass this (or the parent root)
│   ├── model_params/       ← default --item
│   ├── optimizer_state/
│   └── _CHECKPOINT_METADATA
└── doc_state.json
```

Point `-i` at a **step directory** or a **root** that contains numeric step folders (latest is chosen automatically).

## Key maps

| Preset | Use |
|--------|-----|
| `auto` | Detect from tensor names (default) |
| `gemma3` / `gemma2` / `gemma` | Tunix Gemma → HF |
| `llama3` / `llama` | Tunix Llama → HF |
| `qwen` / `qwen2` / `qwen3` | Tunix Qwen → HF |
| `generic` | `kernel`→`weight`, etc. |
| `none` | Keep JAX paths |
| `file.yaml` / `file.json` | Custom mapping |

## Why pass `model=`?

Orbax stores original TPU `NamedSharding`. On CPU those devices do not exist. Providing a live model (or abstract pytree) lets the tool build a **fully replicated** target sharding and reshard on load.

## Publish / develop

```bash
# tests
pytest tests/

# build
python -m build

# upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# upload to PyPI
python -m twine upload dist/*
```

## License

Apache-2.0
