# TileOPs Ascend backend

Experimental TileOPs backend for TileLang-Ascend. The first milestone provides:

- automatic detection of `torch.device("npu")`;
- Ascend 910B2C device facts;
- `AddFwdOp` for equal, contiguous float16/float32 tensors with `alpha=1`;
- a real NPU smoke test.

The backend is intentionally strict. Unsupported shapes, dtypes and parameters raise
`NotImplementedError` instead of silently selecting a CUDA implementation.

## Development installation

Install the compatible TileOPs source tree without replacing the server's PyTorch or
TileLang-Ascend stack, then install this plugin:

```bash
python -m pip install -e /path/to/TileOPs --no-deps --ignore-requires-python
python -m pip install -e backends/tileops-ascend --no-deps
```

The TileOPs upstream metadata currently targets a newer CUDA release stack. The first command
only makes its Python package visible; it does not install CUDA dependencies.

## Smoke test

On the server's current TileLang-Ascend 0.1.4 stack, import this compatibility
plugin before importing TileOPs operators:

```python
import tileops_ascend
from tileops.elementwise import AddFwdOp
```

The explicit import installs compatibility aliases for APIs that current
TileOPs imports before Python entry-point discovery can run (`tvm.tirx` and
`tilelang.jit.JITImpl`). It also registers the `ascend` backend. This bootstrap
can be removed after the server moves to a TileLang version providing those
APIs, or after upstream TileOPs loads backend entry points before importing its
in-tree CUDA kernels.

```bash
pytest -q backends/tileops-ascend/tests
```
