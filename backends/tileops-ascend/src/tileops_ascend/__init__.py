"""Register the Ascend backend with TileOPs."""

import importlib
import sys

import tilelang

# TileOPs 0.1.0 imports ``tvm.tirx`` while TileLang-Ascend 0.1.4 exposes the
# same import-time expression nodes through ``tvm.tir``. CUDA in-tree kernels
# are never selected on this target; the alias only lets the backend-neutral op
# classes load against the older Ascend TVM package.
if "tvm.tirx" not in sys.modules:
    sys.modules["tvm.tirx"] = tilelang.tvm.tir

tilelang_jit = importlib.import_module("tilelang.jit")
if not hasattr(tilelang_jit, "JITImpl"):
    tilelang_jit.JITImpl = tilelang_jit.JITKernel

from tileops.backend import (
    register_detector,
    register_device_facts,
    register_kernel_builder,
)

from tileops_ascend.device import is_ascend_device, read_device_facts
from tileops_ascend.kernels.add import build_add

TARGET = "ascend"

register_detector(target=TARGET, detect=is_ascend_device)
register_device_facts(device_type="npu", read=read_device_facts)
register_kernel_builder(op="AddFwdOp", target=TARGET, build_kernel=build_add)

__all__ = ["TARGET"]
