#!/usr/bin/env bash
# Task C — real NVIDIA Triton + perf_analyzer, end-to-end, on a GPU box (RunPod A6000).
# Produces the `triton` row in edge/results.csv next to `cpp-trt`, on the SAME fp16 engine + A6000.
#
# Pod image: nvcr.io/nvidia/tritonserver:24.10-py3  (Triton 2.51.0, TensorRT 10.5 — matches cpp-trt).
# Driven hands-off: `python3 edge/cpp_server/tools/runpod.py deploy "NVIDIA RTX A6000" <pubkey> SECURE`
# then pipe this over SSH: `ssh ... 'bash -s' < edge/serving/tools/run_triton_bench.sh`.
# Terminate the pod afterwards.
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH=/opt/tritonserver/bin:/usr/src/tensorrt/bin:$PATH   # tritonserver + trtexec (not on PATH by default)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "=== box: $GPU | triton $(cat /opt/tritonserver/TRITON_VERSION 2>/dev/null) | $(trtexec --version 2>/dev/null | head -1) ==="

cd /workspace 2>/dev/null || cd /root
git clone -q https://github.com/HrishiKabra/reefscan.git 2>/dev/null || (cd reefscan && git pull -q)
cd reefscan; REPO=$(pwd)

echo "=== deps (torch/torchvision for ONNX export + F1 parity; tritonclient for the gRPC client) ==="
pip install -q "transformers==4.44.2" "huggingface_hub<0.26" safetensors pyarrow pillow scikit-learn \
  "numpy<2" onnx onnxscript "tritonclient[grpc]" 2>&1 | tail -2
python3 -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
python3 -c "import torchvision" 2>/dev/null || pip install -q torchvision --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install -q "numpy<2" 2>&1 | tail -1
python3 -c "import torch,transformers,numpy; print('torch',torch.__version__,'transformers',transformers.__version__,'numpy',numpy.__version__)"

echo "=== export fp32 ONNX (pixel_values -> logits, dynamic batch) ==="
PYTHONPATH=. python3 - <<'PY'
import os, torch
from edge.model import load_model
from edge.run_rung3 import ONNX_PATH, export_onnx
dev = "cuda" if torch.cuda.is_available() else "cpu"
if not os.path.exists(ONNX_PATH):
    export_onnx(load_model(device=dev), dev)
print("ONNX:", ONNX_PATH, round(os.path.getsize(ONNX_PATH)/1e6), "MB")
PY

echo "=== build fp16 TensorRT plan with trtexec (same min1/opt32/max64 profile as cpp-trt) ==="
MODELDIR=edge/serving/model_repository/reefscan_dinov2/1
mkdir -p $MODELDIR
if [ -f $MODELDIR/model.plan ]; then echo "engine exists — skipping trtexec"; else
  trtexec --onnx=edge/artifacts/dinov2_fp32.onnx --fp16 \
    --minShapes=pixel_values:1x3x224x224 --optShapes=pixel_values:32x3x224x224 --maxShapes=pixel_values:64x3x224x224 \
    --saveEngine=$MODELDIR/model.plan 2>&1 | grep -Ei "error|passed|Engine built" | tail -4
fi
rm -f $MODELDIR/PLACE_ENGINE_HERE.md   # not a model file

echo "=== serve tritonserver ==="
tritonserver --model-repository=$REPO/edge/serving/model_repository >/workspace/triton.log 2>&1 &
for i in $(seq 1 60); do curl -sf localhost:8000/v2/health/ready >/dev/null 2>&1 && { echo "READY ${i}s"; break; }; sleep 2; done
curl -sf localhost:8000/v2/health/ready >/dev/null 2>&1 || { echo "TRITON FAILED"; tail -40 /workspace/triton.log; exit 1; }

echo "=== fetch genuine perf_analyzer (Triton 2.51.0 client tarball; NOT in the base image) ==="
cd /workspace
if [ ! -x /workspace/pa/bin/perf_analyzer ]; then
  wget -q https://github.com/triton-inference-server/server/releases/download/v2.51.0/v2.51.0_ubuntu2204.clients.tar.gz -O clients.tgz
  mkdir -p pa && tar xzf clients.tgz -C pa
fi
export LD_LIBRARY_PATH=/workspace/pa/lib:$LD_LIBRARY_PATH
PA=/workspace/pa/bin/perf_analyzer; $PA --version | head -1
cd $REPO

echo "=== perf_analyzer sweep (native C++ gRPC client; concurrency 1,8,16,32,64) ==="
for C in 1 8 16 32 64; do
  echo "--- concurrency=$C ---"
  $PA -m reefscan_dinov2 -i grpc -u localhost:8001 --shape pixel_values:3,224,224 \
    --concurrency-range ${C}:${C} --measurement-interval 5000 --percentile 95 \
    -f edge/serving/docs/perf_${C}.csv 2>&1 | grep -Ei "Concurrency:|Throughput:|p95 latency" | tail -6 || true
done

echo "=== F1 parity (Python gRPC client) + cross-check curve ==="
PYTHONPATH=. python3 edge/serving/tools/triton_client.py --gpu "$GPU"

echo "=== parse perf_analyzer -> canonical triton row + serving curve ==="
PYTHONPATH=. python3 edge/serving/tools/parse_perf.py --csv-glob 'edge/serving/docs/perf_*.csv' --gpu "$GPU"

echo "=== triton + cpp-trt rows ==="; grep -E 'runtime|triton|cpp-trt' edge/results.csv
echo "=== TRITON BENCH DONE ==="
