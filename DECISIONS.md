# Engineering decisions — serving & inference layer

Honest rationale for the non-obvious choices in the serving/inference work. The theme: measure before
optimizing, keep the comparisons fair, and don't fake infrastructure that can't actually run.

## The CUDA kernel targets the preprocessing *tail*, and it's bandwidth-bound
- **What it fuses:** `uint8 HWC → float32 NCHW → ImageNet-normalize` in one kernel — each `uint8` read
  once, each `float32` written once. The multi-op torch version (`permute → float → /255 → −mean → /std`)
  makes 3–4 separate passes over memory.
- **Why this op:** it's the only preprocessing step that's a pure elementwise/transpose pass (the resize
  needs interpolation and is a different problem). It's the honest place a single fused kernel helps.
- **Why I profile *first* (cell B1):** preprocessing is a small slice of latency. The DINOv2 forward
  dwarfs it, and SAM2's AMG (~seconds) dwarfs *that* — so preprocessing is a tiny share end-to-end. Stating
  that share up front keeps the kernel honest: it's **bandwidth-bound**, the speedup is real but the e2e
  win is small. The deliverable is the *skill* — authoring, binding (`load_inline`), and **verifying**
  (`torch.allclose` vs the multi-op reference) a correct CUDA kernel — not a headline latency number.
- **Why not benchmark vs `torchvision`'s fused normalize:** that op is already a single bandwidth-bound
  kernel, so it's not the thing we're improving on. The fair baseline is the naive multi-op sequence a
  typical pipeline actually writes.

## GPT-4o stays as a column; Qwen2.5-VL is *added*
- The VLM benchmark's job is to show the fine-tuned specialist beats generalist VLMs. GPT-4o is the
  recognizable frontier baseline; deleting it to swap in Qwen would weaken the story.
- Qwen2.5-VL-7B (self-hosted via vLLM) adds a **$0, reproducible** open-model column — anyone can re-run it
  without an OpenAI bill or key. Same prompt, same forced-A/B + logprobs → ECE, same NOAA test split.
- **Fairness details:** coral patches are 224px crops sent **unresized** (Qwen2.5-VL degrades on
  pre-downscaled images; the benchmark was already sending raw base64, so nothing changed). Per-model
  output files (`vlm_benchmark_<model>.json`) so the runs coexist instead of clobbering.

## Triton is deferred to RunPod, not fought in Colab
- Triton needs **a GPU and a Docker daemon on the same host**. Colab is a sandboxed VM with no Docker
  daemon (starting `dockerd` fails on cgroup/GPU-passthrough); a Mac has no NVIDIA GPU. That combination
  simply isn't free.
- Rather than ship a broken "install Docker in Colab" hack, the runnable serving story lives in
  `backend/loadtest.py` and `edge/run_sweep.py` (both Colab/local-runnable), and the *real* Triton path is
  a turnkey `edge/serving/RUNPOD.md` (~15 min, <$1). An engine is TRT-version + GPU-arch specific, so that
  doc is explicit about rebuilding on the box or matching `tritonserver:24.10-py3` (TRT 10.5).

## Load test runs against the real async path, in stub mode
- `backend/loadtest.py` drives the actual `/infer` submit→poll→complete path (not a synthetic endpoint),
  so it measures real queuing. Run in **stub mode** it isolates *serving/queuing overhead*; the JSON records
  `stub: true` + the machine so the numbers are never mistaken for model-compute latency. Real models add
  their latency on top and diverge harder — the honest floor, clearly labeled.

## Part D: `vllm bench serve` primary, async image-sweep fallback
- The brief asks for vLLM's `benchmark_serving`. Its CLI/result-keys move between vLLM versions and its
  stock datasets are text-only (they skip the vision encoder). So the cell runs `vllm bench serve` first,
  and **falls back** to a small async sweep of the *actual image workload* if the CLI yields nothing — so
  the notebook always produces a real p50/p95/p99 + throughput curve on this specific multimodal model,
  regardless of the installed vLLM's exact interface.

## Metrics: percentiles over a window, never a running mean
- p50/p95/p99 are computed over a window of requests; throughput is `count / observed-span`. A running mean
  would hide exactly the tail behavior these metrics exist to expose. p99 is surfaced on the dashboard
  alongside the load-test sweep (labeled with the concurrency it was measured at).
