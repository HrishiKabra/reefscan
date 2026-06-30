# ReefScan-Edge — benchmark results

Same 1,565-image held-out test set for every variant. Latency = warmup + sync-bracketed.
Batch-1 and batched rows are separate (never conflated).

| runtime | precision | device | batch | p50 ms | p95 ms | p99 ms | throughput img/s | peak mem MB | macro-F1 | acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | fp32 | cpu | 1 | 141.31 | 160.99 | 165.70 | 7.0 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cpu | 32 | 3712.90 | 4324.04 | 4361.10 | 8.5 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cuda | 1 | 9.99 | 10.01 | 10.02 | 100.2 | 369 | 0.8853 | 0.8933 |
| pytorch | fp32 | cuda | 32 | 246.39 | 253.12 | 257.96 | 130.2 | 780 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 1 | 8.78 | 9.27 | 9.31 | 112.3 | 363 | 0.8853 | 0.8933 |
| torch.compile | fp32 | cuda | 32 | 239.47 | 241.43 | 242.31 | 134.2 | 603 | 0.8853 | 0.8933 |
