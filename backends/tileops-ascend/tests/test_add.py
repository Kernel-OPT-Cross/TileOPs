import pytest
import torch

import tileops_ascend  # noqa: F401
from tileops.elementwise import AddFwdOp


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="requires an Ascend NPU",
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("explicit_target", [False, True])
def test_add_on_npu(dtype, explicit_target):
    torch.manual_seed(0)
    input_cpu = torch.randn(1024, dtype=dtype)
    other_cpu = torch.randn(1024, dtype=dtype)
    input_npu = input_cpu.npu()
    other_npu = other_cpu.npu()

    op = AddFwdOp(target="ascend" if explicit_target else None)
    output = op(input_npu, other_npu)
    torch.npu.synchronize()

    torch.testing.assert_close(output.cpu(), input_cpu + other_cpu, rtol=1e-3, atol=1e-3)


def test_unsupported_broadcast_is_explicit():
    input_npu = torch.ones(256, dtype=torch.float16).npu()
    other_npu = torch.ones(1, dtype=torch.float16).npu()
    with pytest.raises(NotImplementedError, match="broadcasting"):
        AddFwdOp(target="ascend")(input_npu, other_npu)
