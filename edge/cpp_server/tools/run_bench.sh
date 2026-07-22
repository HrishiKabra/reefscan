#!/usr/bin/env bash
set -eo pipefail
echo "=== box: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
cd /workspace
git clone -q https://github.com/HrishiKabra/reefscan.git 2>/dev/null || (cd reefscan && git pull -q)
cd /workspace/reefscan
pip install -q "transformers==4.44.2" "huggingface_hub<0.26" safetensors pyarrow pillow scikit-learn onnx onnxscript 2>&1 | tail -1

echo "=== engine ==="
python - <<'PY'
import os
from edge.model import load_model
from edge.run_rung3 import ONNX_PATH, export_onnx
from edge.run_rung4 import FP16_PLAN, load_or_build
if not os.path.exists(ONNX_PATH): export_onnx(load_model(device="cuda"), "cuda")
load_or_build("fp16", FP16_PLAN); print("engine ready")
PY

echo "=== build ==="
mkdir -p edge/cpp_server/third_party
[ -f edge/cpp_server/third_party/httplib.h ] || \
  curl -fsSL https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.18.3/httplib.h -o edge/cpp_server/third_party/httplib.h
export PATH=/usr/local/cuda/bin:$PATH
rm -rf edge/cpp_server/build
cmake -S edge/cpp_server -B edge/cpp_server/build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build edge/cpp_server/build -j 2>&1 | tail -3

PYTHONPATH=. python edge/cpp_server/bench/parity_check.py --n 128 >/dev/null 2>&1   # writes input.bin
E=edge/artifacts/dinov2_trt_fp16.plan
ENGINE_PATH=$E ./edge/cpp_server/build/reefscan_server >/workspace/server.log 2>&1 &
SRV=$!
for i in $(seq 1 30); do curl -sf http://localhost:8000/health >/dev/null && break; sleep 1; done

echo "=== NATIVE C++ client (real server throughput) ==="
./edge/cpp_server/build/reefscan_bench_client localhost 8000 edge/cpp_server/_parity/input.bin 128 6000

echo "=== Python httpx client (contrast — client-bound) ==="
PYTHONPATH=. python edge/cpp_server/bench/bench_client.py --url http://localhost:8000 \
  --gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)" --requests 384 2>&1 | grep -E "conc|^ +[0-9]|macro-F1|parity"
kill $SRV 2>/dev/null || true
echo "=== B DONE ==="
