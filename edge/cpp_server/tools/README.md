# RunPod tooling (used to verify the GPU gates hands-off)

- `runpod.py` — REST driver: `python runpod.py {deploy <gpu> <pubkey> <SECURE|COMMUNITY>|status <id>|terminate <id>}` (reads `RUNPOD_API_KEY` from `.env`).
- `run_all_gates.sh` — piped to `bash -s` over SSH on an `nvcr.io/nvidia/pytorch:24.10-py3` pod: builds the fp16 engine + C++ and runs Phase 0-3 gates.
- `run_bench.sh` — same, runs the native C++ load client + Python client (task B).

Flow: deploy → poll `publicIp`+`portMappings[22]` → inject ephemeral SSH key via `dockerStartCmd` → SSH `bash -s < script` → **terminate**. Pins: transformers 4.44.2 on-box, TensorRT 10.5, A6000/A40 secure. Pass ssh `-o` opts INLINE (a shell var mangles them).
