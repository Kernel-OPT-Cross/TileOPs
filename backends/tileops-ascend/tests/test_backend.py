import torch

import tileops_ascend
from tileops.backend import registered_targets


def test_backend_is_registered():
    assert tileops_ascend.TARGET == "ascend"
    assert "ascend" in registered_targets()
    assert tileops_ascend.device.is_ascend_device(torch.device("npu"))
    assert not tileops_ascend.device.is_ascend_device(torch.device("cpu"))


def test_add_is_registered():
    assert registered_targets("AddFwdOp") == ["ascend"]
