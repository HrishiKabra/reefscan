# Triton serving benchmark — tools (task C)

Real NVIDIA Triton serving the fp16 TensorRT engine, load-tested with the official
`perf_analyzer`, producing the `triton` row next to `cpp-trt` in `edge/results.csv`.
Same fp16 engine + same **A6000** as the cpp-trt server, so macro-F1 matches by construction —
the comparison is about the **serving stack**, not the model.

## Files
- `run_triton_bench.sh` — the whole on-box flow (deps → ONNX → `trtexec` fp16 engine → serve
  Triton → `perf_analyzer` sweep → F1 parity → parse row). Piped over SSH to a RunPod pod.
- `triton_client.py` — Python gRPC client. Owns **F1 parity** (perf_analyzer sends random data, so
  it can't check accuracy) → writes `docs/triton_parity.json`. Also a Python-threaded cross-check
  sweep → `docs/triton_pyclient_curve.json` (NOT canonical — throughput is GIL-bound under load).
- `parse_perf.py` — parses perf_analyzer CSVs → `docs/triton_serving_curve.json` + the canonical
  `triton` rows in `edge/results.csv` (macro-F1/acc from `triton_parity.json`).

## Reproduce (hands-off RunPod)
```bash
# 1. deploy a Triton pod (image nvcr.io/nvidia/tritonserver:24.10-py3 = Triton 2.51.0 / TRT 10.5)
python3 edge/cpp_server/tools/runpod.py deploy "NVIDIA RTX A6000" "$(cat ~/.ssh/id_ed25519.pub)" SECURE
# 2. poll status until publicIp + port 22 appear, then:
ssh -i <key> -p <port> root@<ip> 'bash -s' < edge/serving/tools/run_triton_bench.sh
# 3. pull edge/results.csv + edge/serving/docs/* back; commit; TERMINATE the pod.
python3 edge/cpp_server/tools/runpod.py terminate <pod_id>
```

## Gotchas (all cost a re-run — encoded in the script so they don't recur)
- The `tritonserver:24.10-py3` image has **`python3`, not `python`**; `tritonserver` and `trtexec`
  are **not on `PATH`** (add `/opt/tritonserver/bin` + `/usr/src/tensorrt/bin`).
- It ships no `torch`/`torchvision` — installed for the ONNX export + F1 parity.
- **`perf_analyzer` is NOT in the base image**, and the PyPI `perf_analyzer` wheel is a *placeholder
  squat* (v0.1.0, "Placeholder" — no binary). The genuine binary ships in Triton's official client
  tarball `v2.51.0_ubuntu2204.clients.tar.gz` (GitHub release), which the script downloads.
- A TensorRT engine is TRT-version + GPU-arch specific — build it on the box (this script does) and
  keep the box = A6000, Triton = 2.51.0 (TRT 10.5).

## Result (A6000)
perf_analyzer 1→64 concurrency, fp16 engine (full curve in `docs/perf_analyzer.csv`):

| conc | throughput img/s | p50 ms | p95 ms |
|-----:|-----:|-----:|-----:|
| 1  | 199  | 5.01 | 5.49 |
| 8  | 1130 | 6.77 | 8.40 |
| 16 | 1370 | 11.53 | 14.46 |
| 32 | 1490 | 21.14 | 27.16 |
| 64 | 1679 | 37.01 | 45.98 |

**Honest crossover vs the hand-written cpp-trt server:** cpp-trt wins at concurrency-1 (3.6 vs
5.0 ms p50 — no gRPC framing, no forced 1 ms queue-delay wait); Triton wins under load (1490 vs
1240 img/s at concurrency-32, still climbing to ~1.68k at concurrency-64) via its mature dynamic
batcher. perf_analyzer's server-side breakdown attributes the concurrency-1 gap to ~1.2 ms
queue-delay + ~1.1 ms gRPC on top of the ~2.2 ms fp16 kernel. See `edge/RESULTS.md`.
