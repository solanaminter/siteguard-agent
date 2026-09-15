"""Annotation helpers: draw zones, detections, tracks, risk and decisions."""
import cv2
import numpy as np

OK = (46, 204, 113)      # green  (BGR)
VIOLATION = (43, 57, 232)  # red
WARN = (0, 200, 255)       # amber
ZONE_COLOR = (180, 60, 220)


def draw_zones(frame: np.ndarray, zones) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for zone in zones:
        poly = (np.array(zone.polygon, dtype=np.float32) * [w, h]).astype(np.int32)
        cv2.fillPoly(overlay, [poly], ZONE_COLOR)
        cv2.polylines(frame, [poly], True, ZONE_COLOR, 2)
        cv2.putText(frame, f"RESTRICTED: {zone.name}", (poly[0][0] + 6, poly[0][1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ZONE_COLOR, 2)
    return cv2.addWeighted(overlay, 0.12, frame, 0.88, 0)


def annotate(frame: np.ndarray, detections, tracks, zones, banner: str | None = None):
    frame = draw_zones(frame, zones)
    for d in detections:
        x, y, bw, bh = (int(round(v)) for v in d.bbox)
        color = OK if d.hardhat else VIOLATION
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        tag = f"person {d.confidence:.2f}"
        tag += " | HARDHAT" if d.hardhat else " | NO HARDHAT"
        if d.in_restricted_zone:
            tag += f" | IN {d.zone_name}"
        cv2.putText(frame, tag, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        # head ROI marker
        rh = int(bh * 0.28)
        cv2.rectangle(frame, (x, y), (x + bw, y + rh), color, 1)
    for t in tracks:
        x, y, bw, bh = (int(round(v)) for v in t.bbox)
        cv2.putText(frame, f"#{t.track_id}", (x + bw - 44, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    if banner:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(frame, banner, (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WARN, 2)
    return frame
