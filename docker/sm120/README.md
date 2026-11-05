# SM120 experimental environment

This configuration reproduces the environment used to test the
`sm120-experimental` branch on an NVIDIA RTX PRO 5000 Blackwell GPU.

## Host prerequisites

- An NVIDIA GPU reporting compute capability 12.0.
- An NVIDIA driver compatible with CUDA 13.2. The tested driver was 595.99.02.
- Docker Engine with NVIDIA Container Toolkit configured.

The host driver and NVIDIA Container Toolkit are host requirements and are not installed by
this image.

## Build and run

From the repository root:

```bash
docker build -t tileops-sm120:cuda13.2 -f docker/sm120/Dockerfile .
docker run --gpus all --rm -it tileops-sm120:cuda13.2
```

With Docker Compose:

```bash
docker compose -f docker/sm120/compose.yaml build
docker compose -f docker/sm120/compose.yaml run --rm tileops-sm120
```

Inside the container, verify the environment and run the smoke suite:

```bash
python scripts/sm120/verify_environment.py
bash scripts/sm120/run_smoke.sh
```

Run every test file in a separate Python process so a CUDA device-side assertion does not
prevent later files from being tested:

```bash
bash scripts/sm120/run_isolated_tests.sh
```

This branch bypasses TileOPs' architecture filter for SM120. It does not make every kernel
that targets SM90 or another architecture compatible with SM120.
