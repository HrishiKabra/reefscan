# ReefScan-Edge — benchmark results

Same 1,565-image held-out test set for every variant. Latency = warmup + sync-bracketed.
Batch-1 and batched rows are separate (never conflated).

| runtime | precision | device | batch | p50 ms | p95 ms | p99 ms | throughput img/s | peak mem MB | macro-F1 | acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | fp32 | cpu | 1 | 141.31 | 160.99 | 165.70 | 7.0 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cpu | 32 | 3712.90 | 4324.04 | 4361.10 | 8.5 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cuda | 1 | 9.99 | 10.12 | 10.13 | 100.1 | 369 | 0.8853 | 0.8933 |
| pytorch | fp32 | cuda | 32 | 261.72 | 271.34 | 272.28 | 122.1 | 780 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 1 | 9.12 | 9.58 | 9.64 | 109.0 | 363 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 32 | 240.05 | 247.84 | 255.66 | 132.8 | 603 | 0.8853 | 0.8933 |
| onnxruntime | fp32 | cuda | 1 | 4.93 | 5.07 | 5.09 | 202.5 | 361 | 0.8861 | 0.8939 |
| onnxruntime | fp32 | cuda | 32 | 135.48 | 138.28 | 138.88 | 236.7 | 380 | 0.8861 | 0.8939 |
| pytorch | tf32 | cuda | 1 | 8.43 | 8.63 | 9.75 | 118.4 | 369 | 0.8853 | 0.8933 |
| pytorch | tf32 | cuda | 32 | 141.03 | 147.97 | 150.69 | 226.0 | 780 | 0.8853 | 0.8933 |
| onnxruntime | fp16 | cuda | 1 | 3.12 | 3.15 | 3.18 | 320.4 | 356 | 0.8861 | 0.8939 |
| onnxruntime | fp16 | cuda | 32 | 71.57 | 73.32 | 73.84 | 448.4 | 375 | 0.8861 | 0.8939 |
| onnxruntime | int8 | cpu | 1 | 192.73 | 203.69 | 212.74 | 5.2 | — | 0.3992 | 0.6192 |
| onnxruntime | int8 | cpu | 32 | 4178.32 | 4344.09 | 4354.92 | 7.6 | — | 0.3992 | 0.6192 |
