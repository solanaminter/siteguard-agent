"""Synthesize demo video segments from stills via slow Ken Burns pans.

Each segment keeps the pan range small (<=10% of frame width) so the
IoU tracker holds tracks across frames. The violation_corrected segment
starts its pan where violation_nohat's pan ended (reversed direction), so
the track continues across the cut and the agent can withdraw the pending
request when vision sees the hardhat.

Output: assets/annotated/demo_input.mp4 (15 fps, 1280x720)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from siteguard.perception import PersonDetector

RAW = "assets/raw"
OUT = "assets/annotated/demo_input.mp4"
FPS = 15
W, H = 1280, 720

_detector = None


def vertical_anchor(img, scale):
    """Choose the crop's y offset so the topmost detected head stays visible.

    Portrait site photos vertically centered would cut workers' heads out of
    frame -- and the hardhat is the whole story. Anchor instead so the
    highest head top sits ~12% below the frame top.
    """
    global _detector
    ih = img.shape[0]
    sh = int(ih * scale)
    if sh <= H:
        return 0
    if _detector is None:
        _detector = PersonDetector()
    dets = _detector.detect(img)
    if dets:
        top = min(d.bbox[1] for d in dets) * scale
        y = int(top - 0.12 * H)
    else:
        y = (sh - H) // 2
    return max(0, min(y, sh - H))

SEGMENTS = [
    # (image, seconds, label, reverse_pan)
    # seg3 continues seg2's pan end-position (reverse) so the IoU tracker
    # holds the same track across the nohat -> corrected cut; that is what
    # lets the pending approval be WITHDRAWN by later vision.
    ("compliant.webp", 5, "CAM 01", False),
    ("violation_nohat.webp", 5, "CAM 02", False),
    ("violation_corrected.webp", 5, "CAM 02", True),
    ("restricted_zone.webp", 7, "CAM 03", False),
    ("violation_nohat.webp", 5, "CAM 04", True),
]


def pan_frames(img, n_frames, reverse=False):
    ih, iw = img.shape[:2]
    # cover frame: scale so the image covers 1280x720 with 10% pan margin
    scale = max(W / iw, H / ih) * 1.12
    sw, sh = int(iw * scale), int(ih * scale)
    big = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    max_x = sw - W
    y = vertical_anchor(img, scale)
    frames = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        if reverse:
            t = 1 - t
        x = int(t * max_x)
        frames.append(big[y:y + H, x:x + W].copy())
    return frames


def main():
    import os
    os.makedirs("assets/annotated", exist_ok=True)
    all_frames = []
    for fname, secs, label, reverse in SEGMENTS:
        img = cv2.imread(f"{RAW}/{fname}")
        assert img is not None, fname
        frames = pan_frames(img, secs * FPS, reverse=reverse)
        for f in frames:
            cv2.putText(f, label, (24, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2)
            cv2.putText(f, "SITEGUARD LIVE", (W - 260, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        all_frames.extend(frames)
        print(f"segment {fname} ({label}): {len(frames)} frames")
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for f in all_frames:
        vw.write(f)
    vw.release()
    print(f"wrote {OUT}: {len(all_frames)} frames")


if __name__ == "__main__":
    main()
