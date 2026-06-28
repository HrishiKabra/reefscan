"""Active-learning retrain trigger. Phase 8.

Closes the data flywheel: conformal-uncertain segments (set size > 1) are written to
`review_queue`; a human confirms them in /admin/review, which writes to `human_labels`.
This script is the *trigger* — run it (manually or on a cron) to check whether enough new
labels have accumulated to justify a retrain (threshold = 100, to conserve GPU budget), and
if so, export them as a fine-tuning supplement and mark them consumed.

Retraining itself reuses notebooks/01_train_dinov2.ipynb: the exported CSV is added as a
small supplement to the NOAA train split (the human-confirmed crops are pulled from R2/
Supabase Storage by `image_id`), and the model is fine-tuned from the current checkpoint.

Run:  python -m backend.models.retrain_trigger [--threshold 100] [--out data/human_labels_export.csv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ..persistence import supabase

THRESHOLD = 100


def pending_label_count() -> int:
    rows = supabase._select("human_labels", lambda q: q.eq("used_in_training", False))
    return len(rows)


def export_and_mark(out_path: str) -> int:
    rows = supabase._select("human_labels", lambda q: q.eq("used_in_training", False))
    if not rows:
        return 0
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "segment_id", "label"])
        for r in rows:
            w.writerow([r["image_id"], r["segment_id"], r["confirmed_label"]])
    # mark consumed so the next trigger only counts genuinely-new labels
    ids = [r["id"] for r in rows]
    if supabase._client is not None:
        from ..persistence import _sb_call
        _sb_call(lambda: supabase._client.table("human_labels")
                 .update({"used_in_training": True}).in_("id", ids).execute())
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    ap.add_argument("--out", default="data/human_labels_export.csv")
    ap.add_argument("--force", action="store_true", help="export even below threshold")
    a = ap.parse_args()

    if not supabase.enabled:
        print("Supabase not configured (set SUPABASE_URL / SUPABASE_KEY). Nothing to do.")
        return

    n = pending_label_count()
    print(f"new human labels since last retrain: {n} / {a.threshold}")
    if n < a.threshold and not a.force:
        print("below threshold — no retrain yet (conserving GPU budget). "
              "Re-run with --force to export anyway.")
        return

    exported = export_and_mark(a.out)
    print(f"exported {exported} labels -> {a.out}")
    print("next: add this CSV as a supplement in notebooks/01_train_dinov2.ipynb and "
          "fine-tune from the current checkpoint, then push the new stage to the HF Hub.")


if __name__ == "__main__":
    main()
