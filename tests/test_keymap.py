from orbaxport import available_key_maps, make_key_transform, convert
from orbaxport.convert import auto_detect_key_map, _normalize_formats


def test_presets_exist():
    maps = available_key_maps()
    assert "gemma3" in maps
    assert "llama3" in maps
    assert "none" in maps


def test_key_transform_gemma3():
    fn = make_key_transform("gemma3")
    assert fn("embedder.input_embedding") == "model.embed_tokens.weight"
    assert fn("layers.0.mlp.gate_proj.kernel") == "model.layers.0.mlp.gate_proj.weight"
    assert fn("unknown.foo") == "unknown.foo"


def test_auto_detect():
    keys = ["layers.0.attn._query_norm.scale", "embedder.input_embedding"]
    assert auto_detect_key_map(keys) == "gemma3"


def test_normalize_formats():
    assert _normalize_formats("pt,npz") == ["pytorch", "numpy"]
    assert _normalize_formats("mcpack") == ["pack"]
