# ReefScan-Edge — benchmark results

Same 1,565-image held-out test set for every variant. Latency = warmup + sync-bracketed.
Batch-1 and batched rows are separate (never conflated).

| runtime | precision | device | batch | p50 ms | p95 ms | p99 ms | throughput img/s | peak mem MB | macro-F1 | acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | fp32 | cpu | 1 | 141.31 | 160.99 | 165.70 | 7.0 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cpu | 32 | 3712.90 | 4324.04 | 4361.10 | 8.5 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cuda | 1 | 9.99 | 10.23 | 11.27 | 99.6 | 369 | 0.8853 | 0.8933 |
| pytorch | fp32 | cuda | 32 | 241.05 | 247.69 | 248.53 | 132.4 | 780 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 1 | 8.89 | 9.22 | 9.37 | 112.1 | 363 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 32 | 224.79 | 231.41 | 234.35 | 142.1 | 603 | 0.8853 | 0.8933 |
| onnxruntime | fp32 | cuda | 1 | 4.89 | 5.13 | 5.18 | 203.2 | 361 | 0.8861 | 0.8939 |
| onnxruntime | fp32 | cuda | 32 | 126.32 | 127.58 | 128.31 | 253.6 | 380 | 0.8861 | 0.8939 |
