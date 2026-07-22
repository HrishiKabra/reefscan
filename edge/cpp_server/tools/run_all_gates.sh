#!/usr/bin/env bash
set -eo pipefail
echo "=== box: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
nvcc --version | tail -1 || true

cd /workspace
git clone -q https://github.com/HrishiKabra/reefscan.git 2>/dev/null || (cd reefscan && git pull -q)
cd /workspace/reefscan
echo "=== installing edge python deps ==="
# transformers 4.44.2: avoids the container-torch DTensor import (newer transformers needs it) AND
# makes Dinov2 use eager attention (no SDPA -> clean ONNX export).
pip install -q "transformers==4.44.2" "huggingface_hub<0.26" safetensors pyarrow pillow scikit-learn onnx onnxscript 2>&1 | tail -3
python -c "import torch,transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"

echo "=== building the fp16 TensorRT engine (ONNX export -> TRT) ==="
python - <<'PY'
import os
from edge.model import load_model
from edge.run_rung3 import ONNX_PATH, export_onnx
from edge.run_rung4 import FP16_PLAN, load_or_build
if not os.path.exists(ONNX_PATH):
    export_onnx(load_model(device="cuda"), "cuda")
load_or_build("fp16", FP16_PLAN)
print("ENGINE READY:", FP16_PLAN, round(os.path.getsize(FP16_PLAN)/1e6), "MB")
PY

echo "=== fetching cpp-httplib (single header) ==="
mkdir -p edge/cpp_server/third_party
[ -f edge/cpp_server/third_party/httplib.h ] || \
  curl -fsSL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h \
    -o edge/cpp_server/third_party/httplib.h
echo "httplib.h $(wc -l < edge/cpp_server/third_party/httplib.h) lines"

echo "=== building the C++ (all targets) ==="
export PATH=/usr/local/cuda/bin:$PATH
rm -rf edge/cpp_server/build   # clean reconfigure (prior failed CUDA detect cached)
cmake -S edge/cpp_server -B edge/cpp_server/build -DCMAKE_BUILD_TYPE=Release
cmake --build edge/cpp_server/build -j

echo "=== Phase-0 gate: export refs -> run C++ -> compare ==="
PYTHONPATH=. python edge/cpp_server/bench/parity_check.py --n 128
./edge/cpp_server/build/reefscan_infer edge/artifacts/dinov2_trt_fp16.plan \
  edge/cpp_server/_parity/input.bin edge/cpp_server/_parity/cpp_logits.bin 128
PYTHONPATH=. python edge/cpp_server/bench/parity_check.py --check --n 128

echo "=== Phase-1 gate: concurrent producers through the batch queue ==="
E=edge/artifacts/dinov2_trt_fp16.plan; I=edge/cpp_server/_parity/input.bin
./edge/cpp_server/build/reefscan_batch_test $E $I 128 64 32 1000    # normal batching
./edge/cpp_server/build/reefscan_batch_test $E $I 128 128 8 200     # heavy churn: 128 threads, small batch, short delay

echo "=== Phase-2 gate: HTTP server + concurrency sweep -> cpp-trt results row ==="
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ENGINE_PATH=$E ./edge/cpp_server/build/reefscan_server >/workspace/server.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 1; done
PYTHONPATH=. python edge/cpp_server/bench/bench_client.py --url http://localhost:8000 --gpu "$GPU" --requests 384
kill $SRV 2>/dev/null || true

echo "=== Phase-3 gate: fused preprocessing kernel correctness ==="
./edge/cpp_server/build/reefscan_kernel_test

echo "=== updated cpp-trt rows in RESULTS.md ==="
grep -E 'cpp-trt|runtime' edge/RESULTS.md | head -5
echo "=== ALL GATES DONE ==="
