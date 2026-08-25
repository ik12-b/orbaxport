"""TPU-oriented helpers for orbaxport."""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def resolve_path(path: PathLike) -> Path:
    """Resolve local or GCS path. Prefer etils.epath when available."""
    s = str(path)
    if s.startswith("gs://") or s.startswith("gcs://"):
        try:
            from etils import epath
            return epath.Path(s.replace("gcs://", "gs://"))
        except ImportError:
            try:
                import gcsfs
                # return a plain Path-like marker; callers use string for orbax
                return Path(s)
            except ImportError as e:
                raise ImportError(
                    "GCS path requires etils or gcsfs: pip install etils[epath] gcsfs"
                ) from e
    return Path(s).expanduser().resolve()


def is_gcs(path: PathLike) -> bool:
    s = str(path)
    return s.startswith("gs://") or s.startswith("gcs://")


def host_ram_bytes() -> Optional[int]:
    """Best-effort free + total RAM in bytes (Linux)."""
    try:
        meminfo = Path("/proc/meminfo").read_text()
        vals = {}
        for line in meminfo.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].endswith(":"):
                vals[parts[0][:-1]] = int(parts[1]) * 1024  # kB → bytes
        total = vals.get("MemTotal")
        avail = vals.get("MemAvailable") or vals.get("MemFree")
        return avail, total  # type: ignore
    except Exception:
        return None, None


def estimate_checkpoint_bytes(step_dir: Path) -> Dict[str, Any]:
    """
    Estimate on-disk and logical size from Orbax layout without full restore.
    """
    step_dir = Path(step_dir)
    disk = 0
    n_files = 0
    for root, _, files in os.walk(step_dir):
        for f in files:
            p = Path(root) / f
            try:
                disk += p.stat().st_size
                n_files += 1
            except OSError:
                pass

    # Try array_metadatas for logical size (shape * dtype)
    logical = 0
    n_arrays = 0
    for meta_dir in step_dir.rglob("array_metadatas"):
        for mf in meta_dir.iterdir():
            if not mf.is_file():
                continue
            try:
                # Orbax metadata is often binary; also try _METADATA json
                pass
            except Exception:
                pass
    for meta in step_dir.rglob("_METADATA"):
        try:
            data = json.loads(meta.read_text())
            # tree of shapes if present
            def walk(o):
                nonlocal logical, n_arrays
                if isinstance(o, dict):
                    if "shape" in o and "dtype" in o:
                        shape = o["shape"]
                        dtype = str(o["dtype"])
                        item = _dtype_nbytes(dtype)
                        n = int(np.prod(shape)) if shape else 0
                        logical += n * item
                        n_arrays += 1
                    else:
                        for v in o.values():
                            walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
            walk(data)
        except Exception:
            pass

    free, total = host_ram_bytes()
    est = logical or disk
    return {
        "disk_bytes": disk,
        "logical_bytes": logical or None,
        "estimate_bytes": est,
        "n_files": n_files,
        "n_arrays_meta": n_arrays,
        "host_ram_available": free,
        "host_ram_total": total,
        "fits_in_ram": (free is None) or (est < 0.85 * free),
        "human_estimate": _human(est),
        "human_disk": _human(disk),
        "human_ram_free": _human(free) if free else "unknown",
    }


def _dtype_nbytes(dtype: str) -> int:
    d = dtype.lower().replace("numpy.", "").replace("<", "").replace(">", "")
    mapping = {
        "float32": 4, "f4": 4, "float16": 2, "f2": 2, "bfloat16": 2, "bf16": 2,
        "float64": 8, "f8": 8, "int32": 4, "i4": 4, "int64": 8, "i8": 8,
        "int16": 2, "i2": 2, "int8": 1, "i1": 1, "bool": 1, "uint8": 1,
    }
    for k, v in mapping.items():
        if k in d:
            return v
    return 4


def _human(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


def detect_processes(step_dir: Path) -> Dict[str, Any]:
    """Detect multi-process Orbax layout (process_0, process_1, ...)."""
    step_dir = Path(step_dir)
    processes = set()
    for p in step_dir.rglob("ocdbt.process_*"):
        m = re.search(r"ocdbt\.process_(\d+)", p.name)
        if m:
            processes.add(int(m.group(1)))
    for p in step_dir.rglob("array_metadatas"):
        if p.is_dir():
            for c in p.iterdir():
                if c.name.startswith("process_"):
                    try:
                        processes.add(int(c.name.split("_")[1]))
                    except Exception:
                        pass
    return {
        "n_processes": max(len(processes), 1),
        "process_ids": sorted(processes) if processes else [0],
        "multi_process": len(processes) > 1,
    }


def read_sharding_info(step_dir: Path, item: str = "model_params") -> Dict[str, Any]:
    """Best-effort parse of _sharding files and checkpoint metadata."""
    step_dir = Path(step_dir)
    info: Dict[str, Any] = {
        "item": item,
        "has_sharding_file": False,
        "sharding_files": [],
        "mesh_summary": None,
        "notes": [],
    }
    candidates = list((step_dir / item).glob("_sharding")) if (step_dir / item).is_dir() else []
    candidates += list(step_dir.rglob("_sharding"))
    # unique
    seen = set()
    files = []
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            files.append(c)
    info["sharding_files"] = [str(f) for f in files]
    info["has_sharding_file"] = bool(files)

    # _CHECKPOINT_METADATA
    meta_path = step_dir / "_CHECKPOINT_METADATA"
    if meta_path.exists():
        try:
            info["checkpoint_metadata"] = json.loads(meta_path.read_text())
        except Exception:
            info["checkpoint_metadata"] = None

    # Try to load sharding with orbax/jax if available
    try:
        import jax
        from orbax.checkpoint._src.serialization import serialization as ser
        # Just report device count at save-time from string content if text
        for f in files[:3]:
            try:
                raw = f.read_bytes()[:2000]
                text = raw.decode("utf-8", errors="ignore")
                if "mesh" in text.lower() or "PartitionSpec" in text or "NamedSharding" in text:
                    info["notes"].append(f"Sharding file contains mesh/spec text: {f.name}")
                # count device-like patterns
                devs = re.findall(r"TpuDevice|CudaDevice|CpuDevice", text)
                if devs:
                    info["notes"].append(f"Device markers in sharding: {set(devs)}")
            except Exception:
                pass
        info["current_jax_devices"] = [str(d) for d in jax.devices()]
        info["current_backend"] = jax.default_backend()
        info["current_device_count"] = len(jax.devices())
    except Exception as e:
        info["notes"].append(f"JAX/Orbax inspect limited: {e}")

    proc = detect_processes(step_dir)
    info.update(proc)
    return info


def tpu_slice_profile() -> Dict[str, Any]:
    """Collect TPU slice / environment profile."""
    env_keys = [
        "TPU_NAME", "TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES", "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_HOST_BOUNDS", "TPU_TOPOLOGY", "TPU_SKIP_MDS_QUERY", "TF_CONFIG",
        "JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES",
    ]
    env = {k: os.environ[k] for k in env_keys if k in os.environ}
    profile: Dict[str, Any] = {"env": env}
    try:
        import jax
        devices = jax.devices()
        profile["backend"] = jax.default_backend()
        profile["device_count"] = len(devices)
        profile["devices"] = [str(d) for d in devices[:32]]
        kinds = {}
        for d in devices:
            kinds[str(getattr(d, "platform", type(d).__name__))] = kinds.get(
                str(getattr(d, "platform", type(d).__name__)), 0
            ) + 1
        profile["device_kinds"] = kinds
    except Exception as e:
        profile["jax_error"] = str(e)
    return profile


def list_steps_detailed(root: PathLike) -> List[Dict[str, Any]]:
    root = Path(root) if not is_gcs(root) else resolve_path(root)
    steps = []
    try:
        children = list(Path(root).iterdir()) if not is_gcs(root) else list(resolve_path(root).iterdir())
    except Exception:
        return []
    for c in children:
        name = getattr(c, "name", str(c).rstrip("/").split("/")[-1])
        if name.isdigit():
            steps.append({"step": int(name), "path": str(c)})
    steps.sort(key=lambda x: x["step"])
    return steps


def select_step(root: PathLike, step: Union[str, int, None] = "latest") -> Path:
    """
    step: 'latest' | 'best' | int | None
    'best' looks for metrics in doc_state.json / metrics.json if present.
    """
    root_p = resolve_path(root) if is_gcs(root) else Path(root).expanduser().resolve()
    # If already a step dir
    if (Path(str(root_p)) / "model_params").exists() or (Path(str(root_p)) / "_CHECKPOINT_METADATA").exists():
        name = Path(str(root_p)).name
        if name.isdigit() or step in (None, "latest"):
            return Path(str(root_p))

    detailed = list_steps_detailed(root_p)
    if not detailed:
        return Path(str(root_p))

    if step is None or step == "latest":
        return Path(detailed[-1]["path"])
    if isinstance(step, int) or (isinstance(step, str) and step.isdigit()):
        n = int(step)
        for d in detailed:
            if d["step"] == n:
                return Path(d["path"])
        raise FileNotFoundError(f"Step {n} not found under {root}")

    if step == "best":
        # try doc_state.json at root
        for candidate in ("doc_state.json", "metrics.json", "trainer_state.json"):
            p = Path(str(root_p)) / candidate
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    for key in ("best_step", "best_global_step", "global_step", "step"):
                        if key in data:
                            return select_step(root_p, int(data[key]))
                except Exception:
                    pass
        return Path(detailed[-1]["path"])

    raise ValueError(f"Unknown step selector: {step}")


def diff_sharding(step_a: PathLike, step_b: PathLike, item: str = "model_params") -> Dict[str, Any]:
    a = read_sharding_info(Path(step_a), item=item)
    b = read_sharding_info(Path(step_b), item=item)
    return {
        "a": {"path": str(step_a), "n_processes": a.get("n_processes"), "device_count": a.get("current_device_count")},
        "b": {"path": str(step_b), "n_processes": b.get("n_processes"), "device_count": b.get("current_device_count")},
        "process_count_changed": a.get("n_processes") != b.get("n_processes"),
        "sharding_files_a": a.get("sharding_files"),
        "sharding_files_b": b.get("sharding_files"),
        "notes": a.get("notes", []) + b.get("notes", []),
    }


def print_mesh_report(step_dir: PathLike, item: str = "model_params") -> Dict[str, Any]:
    step_dir = Path(step_dir)
    info = read_sharding_info(step_dir, item=item)
    est = estimate_checkpoint_bytes(step_dir)
    profile = tpu_slice_profile()
    report = {"sharding": info, "size": est, "tpu_profile": profile}
    print("=== Mesh / topology report ===")
    print(f"Path: {step_dir}")
    print(f"Item: {item}")
    print(f"Processes: {info.get('n_processes')}  multi={info.get('multi_process')}")
    print(f"Sharding files: {len(info.get('sharding_files') or [])}")
    for n in info.get("notes") or []:
        print(f"  note: {n}")
    print(f"Current JAX backend: {info.get('current_backend')}  devices={info.get('current_device_count')}")
    print(f"Disk size: {est['human_disk']}  estimate: {est['human_estimate']}")
    print(f"Host RAM free: {est['human_ram_free']}  fits_in_ram={est['fits_in_ram']}")
    if not est["fits_in_ram"]:
        print("  WARNING: checkpoint may not fit in RAM — use --layers or --max-shard-size")
    print(f"TPU env: {profile.get('env') or '(none)'}")
    print("=== end report ===")
    return report
