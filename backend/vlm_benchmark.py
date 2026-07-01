"""VLM-vs-specialist benchmark. Phase 8 (+ self-hosted via vLLM).

Zero-shots a frontier VLM on the SAME NOAA test split the specialist is evaluated on, and
compares accuracy + calibration. Works against ANY OpenAI-compatible endpoint:
  - OpenAI GPT-4o (default), or a self-hosted vLLM server via --base-url http://localhost:8000/v1
To keep it cheap and rigorous:
  - images sent at `detail: low` (flat 85 tokens on OpenAI; ignored by vLLM, which sees the
    native-resolution PNG — coral patches are already 224px crops, so NOTHING is downscaled)
  - forced binary choice ("A"=healthy / "B"=bleached), max_tokens=1, temperature=0
  - logprobs of the A/B tokens → a real probability (so we can measure the VLM's ECE)
  - results streamed to a JSONL so a crash never re-bills and runs are resumable

Compares against docs/eval/metrics.json (the specialist) and writes
docs/eval/vlm_benchmark_<model>.json (per-model, so GPT-4o and Qwen results coexist).

Run:  python -m backend.vlm_benchmark [--model gpt-4o] [--limit N] [--workers 6]
      python -m backend.vlm_benchmark --model Qwen/Qwen2.5-VL-7B-Instruct \\
          --base-url http://localhost:8000/v1 --api-key EMPTY --workers 16
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from openai import OpenAI

load_dotenv()
DS = "NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset"
CLASSES = ("healthy", "bleached")
LABEL_MAP = {"CORAL": "healthy", "CORAL_BL": "bleached"}
OUT = Path("docs/eval")

PROMPT = (
    "You are a marine biologist assessing coral health from a cropped underwater photo of a "
    "single coral colony. A bleached colony looks stark white or very pale (it has lost its "
    "symbiotic algae); a healthy colony shows normal pigmentation (browns, greens, tans, blues). "
    "Classify this colony. Answer with exactly one letter: A for healthy, B for bleached."
)


def _load_test() -> list[tuple[bytes, int]]:
    api = HfApi()
    b2l = {}
    for f in api.list_repo_files(DS, repo_type="dataset"):
        p = f.split("/")
        if len(p) >= 3 and f.lower().endswith(".png"):
            b2l[p[-1]] = p[1]
    shards = [f for f in api.list_repo_files(DS, repo_type="dataset", revision="refs/convert/parquet")
              if f.endswith(".parquet") and f.split("/")[-2] == "test"]
    items = []
    for pf in shards:
        path = hf_hub_download(DS, pf, repo_type="dataset", revision="refs/convert/parquet")
        for r in pq.read_table(path, columns=["image"]).column("image").to_pylist():
            cls = LABEL_MAP.get(b2l.get(r["path"]))
            if cls is not None:
                items.append((r["bytes"], CLASSES.index(cls)))
    return items


def classify_one(client, model, img_bytes) -> tuple[int, float]:
    """Return (pred_idx, p_healthy) for one image via forced A/B + logprobs."""
    b64 = base64.b64encode(img_bytes).decode()
    resp = client.chat.completions.create(
        model=model, max_tokens=1, temperature=0, logprobs=True, top_logprobs=12,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "low"}},
        ]}],
    )
    choice = resp.choices[0]
    lp_a = lp_b = None
    try:
        for tok in choice.logprobs.content[0].top_logprobs:
            t = tok.token.strip().upper()
            if t == "A" and lp_a is None:
                lp_a = tok.logprob
            elif t == "B" and lp_b is None:
                lp_b = tok.logprob
    except Exception:  # noqa: BLE001
        pass
    if lp_a is not None and lp_b is not None:
        ea, eb = math.exp(lp_a), math.exp(lp_b)
        p_healthy = ea / (ea + eb)
    else:  # fallback: hard label from the text, no calibrated prob
        letter = (choice.message.content or "").strip().upper()[:1]
        p_healthy = 0.9 if letter == "A" else 0.1
    return (0 if p_healthy >= 0.5 else 1), float(p_healthy)


def ece(probs_healthy, y, bins=10):
    conf = np.maximum(probs_healthy, 1 - probs_healthy)
    pred = (probs_healthy < 0.5).astype(int)  # 0 healthy,1 bleached
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (e.g. a local vLLM server)")
    ap.add_argument("--api-key", default=None, help="defaults to $OPENAI_API_KEY; use 'EMPTY' for local vLLM")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    key = a.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
    client = OpenAI(api_key=key, base_url=a.base_url) if a.base_url else OpenAI(api_key=key)

    # preflight: fail fast + clearly if the endpoint is unreachable (e.g. a not-yet-up vLLM server)
    # instead of grinding through every image with connection errors.
    try:
        client.models.list()
    except Exception as e:  # noqa: BLE001
        where = a.base_url or "the OpenAI API"
        raise SystemExit(
            f"[vlm] cannot reach {where}: {type(e).__name__}: {str(e)[:160]}\n"
            f"      -> is the server up? For vLLM: check logs/vllm.log and "
            f"`curl {(a.base_url or '').rstrip('/')}/models`")

    safe = a.model.replace("/", "_")  # Qwen/Qwen2.5-VL-... -> a valid filename
    items = _load_test()
    if a.limit:
        items = items[:a.limit]
    jsonl = OUT / f"vlm_preds_{safe}.jsonl"
    done = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            r = json.loads(line); done[r["i"]] = r
    print(f"[vlm] {a.model} on {len(items)} test images ({len(done)} cached) ...", flush=True)

    lock = threading.Lock()
    fh = jsonl.open("a")

    def work(i):
        if i in done:
            return done[i]
        import time
        last = None
        for attempt in range(6):
            try:
                pred, ph = classify_one(client, a.model, items[i][0])
                rec = {"i": i, "true": items[i][1], "pred": pred, "p_healthy": ph}
                with lock:
                    fh.write(json.dumps(rec) + "\n"); fh.flush()
                return rec
            except Exception as e:  # noqa: BLE001  (rate limits / transient)
                last = e
                time.sleep(1.5 * (attempt + 1))
        print(f"[vlm] image {i} FAILED after retries: {type(last).__name__}: {str(last)[:160]}", flush=True)
        return None

    results = list(done.values())
    todo = [i for i in range(len(items)) if i not in done]
    n_done = len(done)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, i): i for i in todo}
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)
            n_done += 1
            if n_done % 100 == 0:
                print(f"[vlm] {n_done}/{len(items)}", flush=True)
    fh.close()

    results = [r for r in results if r]
    y = np.array([r["true"] for r in results])
    pred = np.array([r["pred"] for r in results])
    ph = np.array([r["p_healthy"] for r in results])
    acc = float((pred == y).mean())
    from sklearn.metrics import classification_report, confusion_matrix
    rep = classification_report(y, pred, labels=[0, 1], target_names=CLASSES,
                                output_dict=True, zero_division=0)
    out = {
        "model": a.model, "n": len(results),
        "accuracy": acc, "macro_f1": rep["macro avg"]["f1-score"],
        "per_class": {c: rep[c] for c in CLASSES},
        "ece": ece(ph, y),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    # comparison vs specialist
    spec_path = OUT / "metrics.json"
    if spec_path.exists():
        spec = json.loads(spec_path.read_text())
        out["specialist"] = {"accuracy": spec["accuracy"], "macro_f1": spec["macro_f1"], "ece": spec.get("ece")}
    (OUT / f"vlm_benchmark_{safe}.json").write_text(json.dumps(out, indent=2))
    print(f"\n[vlm] {a.model}: acc={acc:.4f} macroF1={out['macro_f1']:.4f} ECE={out['ece']:.4f} (n={len(results)})")
    if "specialist" in out:
        print(f"[vlm] specialist: acc={out['specialist']['accuracy']:.4f} "
              f"macroF1={out['specialist']['macro_f1']:.4f}")
    print(f"[vlm] wrote docs/eval/vlm_benchmark_{safe}.json")


if __name__ == "__main__":
    main()
