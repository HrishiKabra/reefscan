# ReefScan-Edge — Triton serving (production path)

The optimized DINOv2-B classifier (TensorRT fp16 engine from Rung 4) served on **NVIDIA Triton
Inference Server** with **dynamic batching**. This is the "productionize it" path: HTTP/gRPC
endpoints, server-side request batching, and Prometheus metrics — the deployment target the
[serving curve](../docs/serving_curve.png) is really about.

> **Runs on a host with an NVIDIA GPU + Docker.** (It won't run on a Mac or in Colab — Colab
> has the GPU but not Docker; a Mac has neither.) The batch-size / latency / cost tradeoffs are
> reproduced without Docker by `python -m edge.run_sweep` on Colab — that's the runnable version
> of what dynamic batching buys you.

## 1. Drop the engine in
Triton loads a versioned model repo. Copy your Rung-4 fp16 engine in as `model.plan`:
```bash
mkdir -p model_repository/reefscan_dinov2/1
cp ../artifacts/dinov2_trt_fp16.plan model_repository/reefscan_dinov2/1/model.plan
```
**Version match matters:** a TensorRT engine is tied to the TRT version + GPU arch it was built
on. Build the plan with the same TRT the Triton container ships (10.5 → `tritonserver:24.10-py3`)
on the same GPU family, or rebuild it inside the container. Otherwise Triton fails to load it.

## 2. Serve
```bash
docker compose up
# READY when the log shows:  server is alive / model 'reefscan_dinov2' READY
curl -s localhost:8000/v2/health/ready && echo OK
```

## 3. Load-test with perf_analyzer (Rung 6)
`perf_analyzer` sweeps concurrency and reports throughput + latency percentiles under real load —
it exercises the dynamic batcher (concurrent single-image requests get coalesced server-side):
```bash
# from inside the Triton SDK container, or a host with the client installed
perf_analyzer -m reefscan_dinov2 -i grpc -u localhost:8001 \
  --shape pixel_values:3,224,224 \
  --concurrency-range 1:16:2 \
  --percentile 95
```
Read the output as a **latency-throughput curve vs concurrency**: throughput rises with concurrency
until the GPU saturates (~the batched peak in the serving curve), while p95 latency stays flat then
climbs — the same knee, driven by real network + queue instead of a local loop.

## Files
- `model_repository/reefscan_dinov2/config.pbtxt` — TensorRT-plan backend, `max_batch_size=64`,
  `dynamic_batching` (preferred 8/16/32, 1 ms max queue delay), 1 GPU instance.
- `docker-compose.yml` — `tritonserver:24.10-py3` (TRT 10.5), HTTP/gRPC/metrics ports, GPU reservation.
- `model_repository/reefscan_dinov2/1/model.plan` — the engine (you copy it in; large + gitignored).
