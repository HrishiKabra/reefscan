"""Raw dataset label -> ReefScan class mapping.

INITIAL MODEL IS 2-CLASS: healthy | bleached
(Locked from EDA of NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset, which contains only
two health states: CORAL = healthy, CORAL_BL = bleached. No taxonomy, no dead/algae.)

The DB `coral_label` enum still reserves `dead` and `algae_covered` for a future
extension (e.g. ReefNet supplementation) — those are NOT modeled yet, but keeping the
enum values means no DB migration is needed when they arrive. The MODEL head, UI, and
conformal sets are all 2-class for now.
"""
from __future__ import annotations

# The valid target classes for the CURRENT model. 2-class initial model.
# (The DB enum reserves "dead" and "algae_covered" for future extension — see schema.sql.)
CLASSES: tuple[str, ...] = ("healthy", "bleached")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(CLASSES)}

# raw dataset label -> one of CLASSES, or None to explicitly drop.
LABEL_MAPPING: dict[str, str | None] = {
    "CORAL": "healthy",
    "CORAL_BL": "bleached",
}


def map_label(raw_label: str) -> str | None:
    """Return the ReefScan class for a raw dataset label, or None if unmapped/dropped."""
    return LABEL_MAPPING.get(raw_label)
