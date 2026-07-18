"""Phase 0 gate — C++ TRT logit parity vs the Python TRT path (correctness invariant #1).

Same engine, same batch-1, same preprocessed inputs -> the C++ logits must match the Python-TRT
logits (atol 1e-3) and agree on every argmax. A mismatch is a bug in the C++ path, not a result.

Run on the GPU box (needs the edge/ python env + the fp16 engine from run_rung4), from the repo root:

  # 1. export inputs + reference logits
  PYTHONPATH=. python edge/cpp_server/bench/parity_check.py --n 128
  # 2. run the C++ binary it prints, e.g.
  ./edge/cpp_server/build/reefscan_infer edge/artifacts/dinov2_trt_fp16.plan \\
      edge/cpp_server/_parity/input.bin edge/cpp_server/_parity/cpp_logits.bin 128
  # 3. compare
  PYTHONPATH=. python edge/cpp_server/bench/parity_check.py --check --n 128
"""
from __future__ import annotations

import argparse
import os

import numpy as np

DIR = "edge/cpp_server/_parity"
INPUT = os.path.join(DIR, "input.bin")
PY_LOGITS = os.path.join(DIR, "py_logits.npy")
CPP_LOGITS = os.path.join(DIR, "cpp_logits.bin")


def export(n: int) -> None:
    import torch
    from edge.data import load_test
    from edge.run_rung4 import FP16_PLAN, load_or_build, make_predict

    os.makedirs(DIR, exist_ok=True)
    test_x, _ = load_test()
    x = test_x[:n].contiguous()
    x.numpy().astype(np.float32).tofile(INPUT)  # [n,3,224,224] row-major fp32

    engine = load_or_build("fp16", FP16_PLAN)   # same .plan the C++ binary loads
    predict = make_predict(engine)
    with torch.no_grad():  # batch-1 to match the C++ Phase-0 loop exactly (fp16 batch numerics)
        logits = np.concatenate([predict(x[i:i + 1].cuda()).detach().float().cpu().numpy()
                                 for i in range(n)])
    np.save(PY_LOGITS, logits)
    print(f"[parity] wrote {INPUT} ([{n},3,224,224] fp32) + {PY_LOGITS} {logits.shape}")
    print(f"[parity] now run the C++ binary:\n"
          f"    ./edge/cpp_server/build/reefscan_infer <engine.plan> {INPUT} {CPP_LOGITS} {n}\n"
          f"[parity] then: PYTHONPATH=. python {__file__} --check --n {n}")


def check(n: int) -> None:
    py = np.load(PY_LOGITS)[:n]
    cpp = np.fromfile(CPP_LOGITS, dtype=np.float32).reshape(-1, 2)[:n]
    m = min(len(py), len(cpp))
    py, cpp = py[:m], cpp[:m]
    max_abs = float(np.abs(py - cpp).max())
    agree = int((py.argmax(1) == cpp.argmax(1)).sum())
    print(f"[parity] n={m}  max|py - cpp| = {max_abs:.2e}  |  argmax agreement {agree}/{m}")
    ok = max_abs < 1e-3 and agree == m
    print("[parity] PASS — Phase 0 gate met (logit parity)" if ok
          else "[parity] FAIL — investigate the C++ TRT path")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--check", action="store_true", help="compare cpp_logits.bin vs py_logits.npy")
    a = ap.parse_args()
    check(a.n) if a.check else export(a.n)


if __name__ == "__main__":
    main()
