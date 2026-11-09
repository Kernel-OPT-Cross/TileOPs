"""Ascend device detection and TileOPs scheduling facts."""

from __future__ import annotations

import torch

from tileops.backend import DeviceFacts

_ARCH_BY_NAME = {
    "Ascend910B2C": 910202,
}

# 910B2C exposes 192 KiB of UB per vector core. TileOPs calls this field
# ``smem_budget`` for historical CUDA reasons; Ascend builders interpret it as UB capacity.
_UB_BYTES_910B2C = 192 * 1024


def is_ascend_device(device: torch.device) -> bool:
    """Return whether *device* is a torch-npu device."""
    return device.type == "npu"


def read_device_facts(device: torch.device) -> DeviceFacts:
    """Read the subset of Ascend properties used by TileOPs dispatch."""
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("the Ascend backend requires torch_npu") from exc

    index = torch.npu.current_device() if device.index is None else device.index
    name = torch.npu.get_device_name(index)
    if name not in _ARCH_BY_NAME:
        supported = sorted(_ARCH_BY_NAME)
        raise RuntimeError(f"unsupported Ascend device {name!r}; supported: {supported}")
    properties = torch.npu.get_device_properties(index)
    return DeviceFacts(
        arch=_ARCH_BY_NAME[name],
        sm_count=int(properties.cube_core_num),
        smem_budget=_UB_BYTES_910B2C,
    )
