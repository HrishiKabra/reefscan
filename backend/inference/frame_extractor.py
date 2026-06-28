"""Frame extraction. Phase 5.

Image input passes through as a single frame. Video input is split on scene changes via
PySceneDetect (NOT fixed-fps). Both feed the same downstream pipeline — the image/video
distinction collapses (CLAUDE.md). Video decoding degrades gracefully if scenedetect/opencv
are unavailable.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

MAX_VIDEO_FRAMES = 12  # cap representative frames per clip (free-CPU budget)


def extract(data: bytes, kind: str) -> list[Image.Image]:
    """Return a list of frames (PIL RGB) for an image or video upload."""
    if kind == "video":
        return _video_frames(data)
    return [Image.open(io.BytesIO(data)).convert("RGB")]


def _video_frames(data: bytes) -> list[Image.Image]:
    try:
        import cv2  # type: ignore
        from scenedetect import ContentDetector, SceneManager, open_video  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"video support needs scenedetect+opencv: {e}")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        video = open_video(path)
        sm = SceneManager()
        sm.add_detector(ContentDetector())
        sm.detect_scenes(video)
        scenes = sm.get_scene_list()
        cap = cv2.VideoCapture(path)
        frames: list[Image.Image] = []
        # one representative frame per scene (scene start), capped
        targets = [s[0].get_frames() for s in scenes] or [0]
        for fno in targets[:MAX_VIDEO_FRAMES]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, frame = cap.read()
            if ok:
                frames.append(Image.fromarray(frame[:, :, ::-1]))  # BGR -> RGB
        cap.release()
        return frames or [Image.new("RGB", (512, 512))]
    finally:
        Path(path).unlink(missing_ok=True)
