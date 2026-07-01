# Triton on a real GPU box (RunPod / Brev) — productionization appendix

Colab gives a GPU but no Docker daemon; a Mac has neither. Triton needs **a GPU + a Docker daemon on
the same host**, which is a paid cloud box. This is the turnkey path — ~15 min, <$1 — to serve the
TensorRT engine on real NVIDIA Triton and load-test it with `perf_analyzer`. The
[`run_sweep`](../run_sweep.py) / [`backend/loadtest.py`](../../backend/loadtest.py) numbers are the
Colab-runnable stand-ins; this is the genuine article.

## 1. Spin up a box
- **RunPod** → GPU Pod → pick an **L4** or **A10** (matches the engine's build target; ~$0.40–0.80/hr).
  Template: **`runpod/pytorch`** or any CUDA 12.x image with Docker enabled (RunPod pods run Docker).
- **Brev** (`brev.dev`) → `brev create --gpu l4` → `brev shell`. Equivalent; Docker preinstalled.

Either way you land on a host with an NVIDIA GPU **and** a Docker daemon — the combination Colab can't give.

## 2. Get the repo + a matching engine on the box
```bash
git clone https://github.com/HrishiKabra/reefscan.git && cd reefscan/edge/serving
```
A TensorRT engine is **tied to the TRT version + GPU arch it was built on**. Two options:
- **Rebuild on the box** (safest): run `edge/run_rung4.py` there to produce `edge/artifacts/dinov2_trt_fp16.plan`
  with the box's TRT, then copy it in.
- **Reuse a Colab-built plan**: only works if the box's TRT matches (engines built with TRT 10.5 → serve
  with `tritonserver:24.10-py3`, which ships TRT 10.5) *and* the GPU arch matches (L4↔L4).

```bash
mkdir -p model_repository/reefscan_dinov2/1
cp ../artifacts/dinov2_trt_fp16.plan model_repository/reefscan_dinov2/1/model.plan
```

## 3. Serve
```bash
docker compose up            # from edge/serving/  (config.pbtxt + docker-compose.yml already here)
# ready when the log prints:  server is alive  /  model 'reefscan_dinov2' READY
curl -s localhost:8000/v2/health/ready && echo READY
```

## 4. Load-test with perf_analyzer (the Rung-6 measurement, for real)
```bash
docker run --rm --network host nvcr.io/nvidia/tritonserver:24.10-py3-sdk \
  perf_analyzer -m reefscan_dinov2 -i grpc -u localhost:8001 \
    --shape pixel_values:3,224,224 \
    --concurrency-range 1:16:2 \
    --percentile 95
```
Read the output as **throughput + p95 latency vs concurrency**: throughput climbs until the GPU
saturates (≈ the batched peak in the [serving curve](../docs/serving_curve.png)) while p95 stays flat
then rises — the same knee, now driven by real network + Triton's dynamic batcher instead of a local loop.

## 5. Tear down
Stop the pod/instance so you stop paying. Total cost for the run is well under $1.

---
Files used here: [`docker-compose.yml`](docker-compose.yml) (pins `tritonserver:24.10-py3` = TRT 10.5),
[`model_repository/reefscan_dinov2/config.pbtxt`](model_repository/reefscan_dinov2/config.pbtxt)
(`tensorrt_plan`, `max_batch_size=64`, `dynamic_batching` preferred 8/16/32).
