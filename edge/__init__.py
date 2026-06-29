"""ReefScan-Edge — inference-optimization ladder over the trained DINOv2-B classifier.

The benchmark harness (harness.py) is the spine: every rung registers a (runtime, precision,
batch) variant via benchmark() and appends to results.csv / RESULTS.md. See docs/V2_SPEC.md.
"""
