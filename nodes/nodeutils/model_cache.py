"""
Model caching system for ComfyUI-MeshCraft (Improved)

Provides a global, device-safe model cache to avoid reloading large models on every workflow run.
Features:
- Deterministic cache keys (JSON-based)
- Device-aware reloading
- Optional auto-offload to CPU to prevent VRAM leaks
- Clear, concise debug logging
"""

import hashlib
import json
import torch
import gc
from typing import Optional, Dict, Any

# Global cache (persists across workflow runs)
# Each entry: {"model": model_obj, "device": "cuda:0" or "cpu"}
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}

# Optional VRAM safeguard (in MB)
_MAX_CACHED_VRAM_MB = 16000  # tweak if needed (e.g., 16 GB)


def get_cache_key(model_path: str, **kwargs) -> str:
    """
    Generate deterministic cache key based on model path and config.
    """
    key_str = json.dumps(
        {"path": model_path, "kwargs": kwargs},
        sort_keys=True,
        default=str,
    )
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def _current_device() -> str:
    import comfy.model_management as mm
    return str(mm.get_torch_device())


def _maybe_offload_if_vram_high():
    """Prevent excessive VRAM use by offloading least-recently-used models to CPU."""
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_reserved() / (1024**2)
    if used > _MAX_CACHED_VRAM_MB:
        print(f"⚠️  VRAM usage high ({used:.0f} MB) — offloading cached models to CPU...")
        for k, entry in _MODEL_CACHE.items():
            model = entry["model"]
            try:
                if hasattr(model, "to"):
                    model.to("cpu")
                    _MODEL_CACHE[k]["device"] = "cpu"
            except Exception as e:
                print(f"  Skipped offloading {k[:8]}: {e}")
        torch.cuda.empty_cache()
        gc.collect()


def get_cached_model(cache_key: str) -> Optional[Any]:
    """
    Retrieve model from cache and move to active GPU if needed.
    """
    entry = _MODEL_CACHE.get(cache_key)
    if not entry:
        print(f"🔍 Cache miss for key {cache_key[:12]}")
        return None

    model, cached_device = entry["model"], entry["device"]
    current_device = _current_device()

    print(f"🔍 Cache hit for key {cache_key[:12]} ({type(model).__name__}) [cached on {cached_device}]")

    # Move to correct device if necessary
    if cached_device != current_device and hasattr(model, "to"):
        try:
            model = model.to(current_device)
            entry["model"] = model
            entry["device"] = current_device
            print(f"✓ Moved cached model from {cached_device} → {current_device}")
        except Exception as e:
            print(f"⚠️  Failed to move cached model: {e}")

    return model


def cache_model(cache_key: str, model: Any) -> None:
    """
    Store model in cache (GPU if available, else CPU).
    """
    device = _current_device()
    if hasattr(model, "to"):
        try:
            model = model.to(device)
            print(f"✓ Model moved to {device} before caching")
        except Exception as e:
            print(f"⚠️  Could not move model to {device}: {e}")

    _MODEL_CACHE[cache_key] = {"model": model, "device": device}
    print(f"💾 Cached model {type(model).__name__} [{cache_key[:12]}...] on {device} "
          f"(total {len(_MODEL_CACHE)} models)")

    _maybe_offload_if_vram_high()


def clear_cache() -> None:
    """
    Clear all cached models and free GPU memory.
    """
    _MODEL_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Cleared model cache and freed GPU memory.")


def get_cache_info() -> Dict[str, Any]:
    """
    Return summary of current cache state.
    """
    return {
        "num_cached_models": len(_MODEL_CACHE),
        "entries": [
            {"key": k[:12], "device": v["device"], "type": type(v["model"]).__name__}
            for k, v in _MODEL_CACHE.items()
        ],
    }