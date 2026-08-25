# Changelog

## 1.1.0

- Multi-format export: safetensors, pytorch, numpy, msgpack, pickle
- `mcpack` / pack bundle (zip + manifest)
- Auto key-map detection
- Split fused QKV/KV, optional Linear transpose
- Partial layer export, sharded safetensors, HF folder mode
- LoRA-only adapter export
- Compare utility, dry-run, list-keys, provenance metadata
- Safe writes via /tmp + move

## 1.0.0

- Initial public release: Orbax/Tunix → SafeTensors with TPU→CPU resharding helpers
