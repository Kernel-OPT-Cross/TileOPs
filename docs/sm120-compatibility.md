# Experimental SM120 compatibility

This branch adds an experimental fallback that permits an SM120 device to enter TileOPs'
normal implementation selection and kernel compilation paths. The change is deliberately
small: it relaxes two architecture checks in `kernel_base.py`. It does not rewrite kernels
or claim that kernels designed for SM90 are automatically compatible with SM120.

## Tested environment

| Component | Version |
| --- | --- |
| GPU | NVIDIA RTX PRO 5000 Blackwell |
| Compute capability | 12.0 (SM120) |
| NVIDIA driver | 595.99.02 |
| CUDA development image | `nvidia/cuda:13.2.1-devel-ubuntu22.04` |
| CUDA compiler | 13.2.78 |
| Python | 3.12.16 |
| PyTorch | 2.13.0 |
| TileLang | 0.1.12 |
| TileOPs base | v0.1.0 (`f7e03a75943550060729167a54f8bfe2f9f242d3`) |

## Isolated full-suite result

Each of the 175 test files was launched in a separate process. This isolation is important:
a CUDA device-side assertion poisons its process and otherwise creates misleading cascading
errors in later tests.

- 127 test files passed completely.
- 48 test files contained a failure or error.
- 5,276 test cases passed.
- 899 test cases failed.
- 98 test cases errored.
- 833 test cases were skipped.

These results show that a substantial part of TileOPs runs on this SM120 system, but the
branch is not a complete SM120 port. See `scripts/sm120/run_isolated_tests.sh` for the test
strategy and `docker/sm120/README.md` for reproduction instructions.

## Known limitation

Do not treat successful architecture dispatch as proof of kernel compatibility. Kernels may
still fail during compilation or execution. Keep unrelated workloads isolated in separate
processes when evaluating new operators.
