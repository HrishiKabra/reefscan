"""ReefScan data pipeline (Phase 2): imagefolder patch Dataset, transforms, EDA.

CoralPatchDataset (ImageFolder) is the loader for the current 2-class NOAA dataset.
backend.data.split is retained for FUTURE point-annotated sources (not used here).
"""
from .label_mapping import CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, LABEL_MAPPING, map_label
from .patch_dataset import CoralPatchDataset, scan_imagefolder

__all__ = [
    "CoralPatchDataset",
    "scan_imagefolder",
    "CLASSES",
    "CLASS_TO_IDX",
    "IDX_TO_CLASS",
    "LABEL_MAPPING",
    "map_label",
]
