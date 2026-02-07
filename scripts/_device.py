from __future__ import annotations

def pick_best_device() -> str:
    """
    Returns: 'gpu' (CUDA), 'mps' (Apple Silicon), or 'cpu'
    """
    try:
        import torch
    except Exception:
        return "cpu"

    # NVIDIA CUDA
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "gpu"

    # Apple Silicon (Metal Performance Shaders)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
