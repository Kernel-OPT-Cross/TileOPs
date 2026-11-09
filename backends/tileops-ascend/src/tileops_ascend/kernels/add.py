"""Developer-mode TileLang-Ascend implementation of TileOPs AddFwdOp."""

import math
from collections.abc import Callable

import torch

import tilelang
import tilelang.language as T
from tileops.backend import TensorSpec

_BLOCK_ELEMENTS = 256
_SUPPORTED_DTYPES = {torch.float16: "float16", torch.float32: "float32"}
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=_PASS_CONFIGS)
def _add_kernel(numel: int, dtype: str):
    block_count = numel // _BLOCK_ELEMENTS
    vector_elements = _BLOCK_ELEMENTS // 2

    @T.prim_func
    def main(
        a: T.Tensor((numel,), dtype),
        b: T.Tensor((numel,), dtype),
        output: T.Tensor((numel,), dtype),
    ):
        with T.Kernel(block_count, is_npu=True) as (cid, vid):
            a_shared = T.alloc_shared((vector_elements,), dtype)
            b_shared = T.alloc_shared((vector_elements,), dtype)
            output_shared = T.alloc_shared((vector_elements,), dtype)
            offset = cid * _BLOCK_ELEMENTS + vid * vector_elements

            T.copy(a[offset], a_shared)
            T.copy(b[offset], b_shared)
            for i in T.Parallel(vector_elements):
                output_shared[i] = a_shared[i] + b_shared[i]
            T.copy(output_shared, output[offset])

    return main


class _AddKernel:
    """Shape-specialized callable returned to TileOPs."""

    def __init__(self, shape: tuple[int, ...], dtype: torch.dtype):
        self.shape = shape
        self.numel = math.prod(shape)
        self.compiled = _add_kernel(self.numel, _SUPPORTED_DTYPES[dtype])
        self.config = {"block_elements": _BLOCK_ELEMENTS, "mode": "developer"}

    def __call__(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        flat = self.compiled(input.view(-1), other.view(-1))
        return flat.view(self.shape)


def build_add(
    input: TensorSpec,
    other: TensorSpec,
    *,
    alpha: int | float,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the supported equal-shape subset of TileOPs AddFwdOp."""
    if input.device.type != "npu" or other.device.type != "npu":
        raise NotImplementedError("the Ascend Add backend only accepts NPU tensors")
    if input.device != other.device:
        raise NotImplementedError("the Ascend Add backend requires both tensors on one NPU")
    if input.shape != other.shape:
        raise NotImplementedError("the initial Ascend Add backend does not support broadcasting")
    if input.dtype != other.dtype or input.dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype) for dtype in sorted(_SUPPORTED_DTYPES, key=str))
        raise NotImplementedError(f"the initial Ascend Add backend supports only {supported}")
    if alpha != 1:
        raise NotImplementedError("the initial Ascend Add backend supports only alpha=1")
    numel = math.prod(input.shape)
    if numel == 0 or numel % _BLOCK_ELEMENTS:
        raise NotImplementedError(
            f"the initial Ascend Add backend requires a positive element count divisible by "
            f"{_BLOCK_ELEMENTS}, got {numel}"
        )
    return _AddKernel(input.shape, input.dtype)
