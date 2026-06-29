# ReefScan-Edge — benchmark results

Same 1,565-image held-out test set for every variant. Latency = warmup + sync-bracketed.
Batch-1 and batched rows are separate (never conflated).

| runtime | precision | device | batch | p50 ms | p95 ms | p99 ms | throughput img/s | peak mem MB | macro-F1 | acc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch | fp32 | cpu | 1 | 141.31 | 160.99 | 165.70 | 7.0 | — | 0.8874 | 0.8952 |
| pytorch | fp32 | cpu | 32 | 3712.90 | 4324.04 | 4361.10 | 8.5 | — | 0.8874 | 0.8952 |
