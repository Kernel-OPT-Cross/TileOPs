from __future__ import annotations

import platform

import tilelang
import tileops
import torch


def version_of(module: object) -> str:
    return str(getattr(module, "__version__", "unknown"))


print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"TileLang: {version_of(tilelang)}")
print(f"TileOPs: {version_of(tileops)}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the container")

major, minor = torch.cuda.get_device_capability()
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute capability: {major}.{minor}")

if (major, minor) != (12, 0):
    raise SystemExit(f"Expected an SM120 GPU, found SM{major}{minor}")
