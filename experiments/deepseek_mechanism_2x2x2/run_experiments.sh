#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f gpu_profile.json ]]; then
  python3 benchmark_gpu.py
fi
python3 run_experiments.py "$@"
