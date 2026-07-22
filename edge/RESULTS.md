# ReefScan-Edge — benchmark results

Same 1,565-image held-out test set for every variant. Latency = warmup + sync-bracketed.
Batch-1 and batched rows are separate (never conflated).

| runtime | precision | device | batch | p50 ms | p95 ms | p99 ms | throughput img/s | peak mem MB | macro-F1 | acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | fp32 | cpu | 1 | 141.31 | 160.99 | 165.70 | 7.0 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cpu | 32 | 3712.90 | 4324.04 | 4361.10 | 8.5 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cuda | 1 | 9.98 | 10.17 | 10.29 | 100.1 | 369 | 0.8853 | 0.8933 |
| pytorch | fp32 | cuda | 32 | 247.53 | 257.49 | 259.76 | 129.2 | 780 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 1 | 8.83 | 9.03 | 9.23 | 113.1 | 363 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 32 | 236.19 | 240.67 | 241.55 | 136.2 | 603 | 0.8853 | 0.8933 |
| onnxruntime | fp32 | cuda | 1 | 4.79 | 4.94 | 5.02 | 208.3 | 361 | 0.8861 | 0.8939 |
| onnxruntime | fp32 | cuda | 32 | 130.35 | 132.64 | 133.03 | 245.6 | 380 | 0.8861 | 0.8939 |
| pytorch | tf32 | cuda | 1 | 8.26 | 8.64 | 10.64 | 119.5 | 369 | 0.8853 | 0.8933 |
| pytorch | tf32 | cuda | 32 | 136.32 | 142.84 | 143.72 | 233.8 | 780 | 0.8853 | 0.8933 |
| onnxruntime | fp16 | cuda | 1 | 3.09 | 3.14 | 3.19 | 323.3 | 356 | 0.8861 | 0.8939 |
| onnxruntime | fp16 | cuda | 32 | 66.24 | 67.39 | 67.98 | 483.2 | 375 | 0.8861 | 0.8939 |
| onnxruntime | int8 | cpu | 1 | 186.11 | 192.30 | 192.66 | 5.3 | — | 0.3992 | 0.6192 |
| onnxruntime | int8 | cpu | 32 | 4050.36 | 4446.66 | 4464.14 | 7.7 | — | 0.3992 | 0.6192 |
| tensorrt | fp16 | cuda | 1 | 2.24 | 2.31 | 2.39 | 443.4 | 361 | 0.8888 | 0.8965 |
| tensorrt | fp16 | cuda | 32 | 34.69 | 35.78 | 36.23 | 923.6 | 380 | 0.8888 | 0.8965 |
| tensorrt | int8 | cuda | 1 | 2.29 | 2.39 | 2.45 | 434.3 | 361 | 0.8840 | 0.8920 |
| tensorrt | int8 | cuda | 32 | 35.40 | 36.69 | 37.14 | 903.5 | 380 | 0.8840 | 0.8920 |
| cpp-trt † | fp16 | cuda | 1 | 3.61 | 4.16 | 4.81 | 274.5 | — | 0.8881 | 0.8958 |
| cpp-trt † | fp16 | cuda | 32 | 26.08 | 31.47 | 33.20 | 1240.8 | — | 0.8881 | 0.8958 |

† **cpp-trt** = the hand-written C++ server (`edge/cpp_server/`), measured with the **native C++ load client**. Latency is **end-to-end HTTP** (network + dynamic-batch queue + TensorRT) and the `batch` column is **client concurrency** (the server batches internally), so these rows aren't directly comparable to the in-process runtime rows above (e.g. tensorrt fp16's 2.24 ms is bare kernel time). It peaks **~1.3k req/s** and does **3.6 ms p50 @ concurrency-1** (after a `TCP_NODELAY` fix that removed a ~40 ms Nagle stall). See `edge/cpp_server/DECISIONS.md`.
