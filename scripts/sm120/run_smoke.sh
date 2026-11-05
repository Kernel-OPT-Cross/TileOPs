#!/usr/bin/env bash
set -euo pipefail

python scripts/sm120/verify_environment.py
python -m pytest -ra -q tests -m smoke
