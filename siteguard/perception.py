"""OpenCV 5 perception layer: person detection, hardhat segmentation,
zone intrusion, optical flow, and multi-object tracking.

All vision runs locally with OpenCV 5 — no hosted vision APIs.
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import PerceptionConfig


@dataclass
class Detection:
    bbox: tuple  # (x, y, w, h) in pixels
    confidence: float = 1.0
    hardhat: bool = False
    hardhat_confidence: float = 0.0
    in_restricted_zone: bool = False
    zone_name: str | None = None
    motion: float = 0.0


@dataclass
class Track:
    track_id: int
    bbox: tuple
    frames_seen: int = 0
    hardhat_votes: list = field(default_factory=list)  # recent bool votes
    zone_frames: int = 0  # consecutive frames inside a restricted zone
    violation_frames: int = 0  # consecutive frames with any violation
    motion: float = 0.0
    last_seen_frame: int = 0

    @property
    def hardhat_ratio(self) -> float:
        if not self.hardhat_votes:
            return 0.0
        return sum(1 for v in self.hardhat_votes if v) / len(self.hardhat_votes)


class PersonDetector:
    """YOLOX-s person detector from the official OpenCV Model Zoo,
    run through OpenCV 5's dnn module.

    Notes on the OpenCV 5 migration: cv2.HOGDescriptor and the Darknet/Caffe
    model importers were removed from the OpenCV 5 Python bindings, so person
    detection uses cv2.dnn.readNetFromONNX + cv2.dnn.NMSBoxesBatched, all
    inside OpenCV 5. Weights: OpenCV Zoo's YOLOX-s
    (object_detection_yolox_2022nov.onnx, Apache 2.0, tuned for the OpenCV
    dnn module); see assets/models/SHA256SUMS and scripts/download_models.sh.

    Preprocessing follows the Zoo reference: BGR->RGB, letterbox to 640x640
    with 114.0 fill, NCHW float32, no mean/std normalization. Postprocessing
    decodes the anchor-free head (strides 8/16/32) and multiplies objectness
    by class scores before batched NMS.
    """

    PERSON_CLASS = 0  # COCO class id for "person"
    STRIDES = (8, 16, 32)

    def __init__(self, model_path: str | None = None,
                 conf_thresh: float = 0.35, nms_thresh: float = 0.5,
                 input_size: int = 640):
        if not model_path:
            model_path = str(
                Path(__file__).resolve().parent.parent
                / "assets" / "models" / "yolox_s_2022nov.onnx"
            )
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net = net
        self._grids, self._strides = self._make_grids(input_size)

    def _forward(self, blob: np.ndarray) -> np.ndarray:
        """Run the ONNX graph. Overridable so alternate runtimes
        (e.g. ONNX Runtime with a CUDA execution provider) can be
        injected without changing preprocessing/postprocessing."""
        self.net.setInput(blob)
        return self.net.forward()[0]  # (8400, 85): [cx,cy,bw,bh, obj, 80 cls]

    @classmethod
    def _make_grids(cls, input_size: int):
        grids, strides = [], []
        for s in cls.STRIDES:
            n = input_size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(1, -1, 2))
            strides.append(np.full((1, n * n, 1), s, dtype=np.float32))
        grid = np.concatenate(grids, axis=1).astype(np.float32)[0]
        stride = np.concatenate(strides, axis=1).astype(np.float32)[0, :, 0]
        return grid, stride

    def _letterbox(self, rgb: np.ndarray):
        size = self.input_size
        padded = np.full((size, size, 3), 114.0, dtype=np.float32)
        ratio = min(size / rgb.shape[0], size / rgb.shape[1])
        resized = cv2.resize(
            rgb, (int(rgb.shape[1] * ratio), int(rgb.shape[0] * ratio))
        ).astype(np.float32)
        padded[: resized.shape[0], : resized.shape[1]] = resized
        return padded, ratio

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        padded, ratio = self._letterbox(rgb)
        blob = np.transpose(padded, (2, 0, 1))[np.newaxis].astype(np.float32)
        out = self._forward(blob)

        dets = out.copy()
        dets[:, 0:2] = (dets[:, 0:2] + self._grids) * self._strides[:, None]
        dets[:, 2:4] = np.exp(np.clip(dets[:, 2:4], -10, 10)) * self._strides[:, None]

        boxes_xywh = np.empty_like(dets[:, :4])
        boxes_xywh[:, 0] = dets[:, 0] - dets[:, 2] / 2.0
        boxes_xywh[:, 1] = dets[:, 1] - dets[:, 3] / 2.0
        boxes_xywh[:, 2:4] = dets[:, 2:4]

        scores = dets[:, 4:5] * dets[:, 5:]  # objectness * class score
        max_scores = np.amax(scores, axis=1)
        max_ids = np.argmax(scores, axis=1)

        keep = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(), max_scores.tolist(), max_ids.tolist(),
            self.conf_thresh, self.nms_thresh,
        )
        out_dets = []
        for k in np.array(keep).flatten():
            k = int(k)
            if int(max_ids[k]) != self.PERSON_CLASS:
                continue
            x, y, bw, bh = boxes_xywh[k] / ratio
            x, y = max(0.0, x), max(0.0, y)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            out_dets.append(
                Detection(bbox=(x, y, bw, bh), confidence=float(max_scores[k])))
        return out_dets


# HSV ranges for saturated hardhat colors. Yellow starts at H=20: brown dirt
# (the main background confounder) peaks below H=20, yellow hardhats at 20-35.
_HARDHAT_SAT_RANGES = [
    ((20, 90, 90), (40, 255, 255)),    # yellow
    ((8, 100, 100), (20, 255, 255)),   # orange
    ((0, 100, 100), (8, 255, 255)),    # red (low)
    ((160, 100, 100), (180, 255, 255)),  # red (high)
    ((90, 80, 80), (115, 255, 255)),   # blue
]
_HARDHAT_YELLOW_RANGE = ((20, 90, 90), (40, 255, 255))
# White / light-gray hardhats. Kept separate because bright backgrounds
# (sky, concrete) also match; white needs the extra texture/shape checks.
_HARDHAT_WHITE_RANGE = ((0, 0, 180), (180, 50, 255))


class HardhatClassifier:
    """Decides hardhat vs. bare head from the head ROI.

    Pipeline (all classical OpenCV):
      1. Head ROI = top 15% of the person box, inset 15% on each side
         (hardhats sit centered on the head; insets reject background).
      2. HSV segmentation into saturated helmet colors, yellow-only, white.
      3. Three detection paths (OR):
         A. Shape: a compact, solid, dome-like helmet-colored blob that does
            not span the full ROI width (sky/dirt/walls do).
         B. Yellow dominance: yellow-hue coverage >= 10% — catches close-up
            yellow hardhats that fill the ROI (dirt/sky have ~zero yellow).
         C. White: a dome-like white blob with brim/ridge edge texture
            (smooth sky has none).
      4. Bare heads fail all three: background blobs span the ROI or lack
         helmet hues; faces/hair don't match.

    Known limitation: a white hardhat against a bright white sky merges into
    one background blob and is missed (see assets/raw/whitehat_sky.webp,
    kept as an explicit failure probe in the eval set).
    """

    def __init__(self, cfg: PerceptionConfig):
        self.cfg = cfg

    @staticmethod
    def _dome_like(mask: np.ndarray) -> bool:
        """True if the mask holds a compact, solid, dome-shaped blob."""
        rh, rw = mask.shape
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < 0.02 * rh * rw or area > 0.65 * rh * rw:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            if bx <= 1 and bx + bw >= rw - 1:
                continue  # spans the full ROI width -> background
            hull_area = cv2.contourArea(cv2.convexHull(c))
            if area / max(hull_area, 1) < 0.82:
                continue  # irregular mound/foliage, not a dome
            if len(c) >= 5:
                (_, _), (ew, eh), _ = cv2.fitEllipse(c)
                aspect = max(ew, eh) / max(min(ew, eh), 1)
            else:
                aspect = bw / max(bh, 1)
            if not 1.1 <= aspect <= 3.5:
                continue
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            if 0.10 * rw <= cx <= 0.90 * rw and cy <= 0.75 * rh:
                return True
        return False

    def classify(self, frame: np.ndarray, bbox: tuple) -> tuple[bool, float]:
        x, y, w, h = (int(round(v)) for v in bbox)
        H, W = frame.shape[:2]
        roi_h = max(8, int(h * 0.15))
        ix = int(w * 0.15)
        x0, y0 = max(0, x + ix), max(0, y)
        x1, y1 = min(W, x + w - ix), min(H, y + roi_h)
        head = frame[y0:y1, x0:x1]
        if head.size == 0:
            return False, 0.0
        hsv = cv2.cvtColor(head, cv2.COLOR_BGR2HSV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        sat_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _HARDHAT_SAT_RANGES:
            sat_mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)

        yellow_mask = cv2.inRange(
            hsv, np.array(_HARDHAT_YELLOW_RANGE[0]),
            np.array(_HARDHAT_YELLOW_RANGE[1]))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

        white_mask = cv2.inRange(
            hsv, np.array(_HARDHAT_WHITE_RANGE[0]),
            np.array(_HARDHAT_WHITE_RANGE[1]))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        sat_cov = float(sat_mask.mean() / 255.0)
        yellow_cov = float(yellow_mask.mean() / 255.0)
        white_cov = float(white_mask.mean() / 255.0)

        path_a = self._dome_like(sat_mask)
        path_b = yellow_cov >= 0.10
        path_c = False
        if white_cov >= 0.10 and self._dome_like(white_mask):
            gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 150)
            white_px = max(int((white_mask > 0).sum()), 1)
            edge_px = int(np.logical_and(edges > 0, white_mask > 0).sum())
            path_c = (edge_px / white_px) > 0.015

        hardhat = bool(path_a or path_b or path_c)
        cov = max(sat_cov, yellow_cov, white_cov)
        conf = min(1.0, cov / max(self.cfg.hardhat_min_coverage, 1e-6))
        return hardhat, conf


class ZoneChecker:
    """Point-in-polygon restricted-zone tests."""

    def __init__(self, zones):
        self.zones = zones

    def check(self, frame_shape, bbox: tuple):
        h, w = frame_shape[:2]
        x, y, bw, bh = bbox
        cx, cy = (x + bw / 2) / w, (y + bh / 2) / h  # foot-ish center in 0..1
        for zone in self.zones:
            poly = (np.array(zone.polygon, dtype=np.float32) * [w, h]).astype(np.int32)
            if cv2.pointPolygonTest(poly, (cx * w, cy * h), False) >= 0 and zone.restricted:
                return True, zone.name
        return False, None


class MotionEstimator:
    """Dense Farneback optical flow -> per-frame mean motion magnitude."""

    def __init__(self):
        self.prev_gray = None

    def estimate(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            # (Re)set on the first frame and whenever the resolution changes
            # (e.g. a pipeline reused across stills of different sizes).
            self.prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag = np.linalg.norm(flow, axis=2)
        self.prev_gray = gray
        return float(np.percentile(mag, 95))


class CentroidTracker:
    """Lightweight IoU-based multi-object tracker over detections."""

    def __init__(self, max_missed: int = 10, iou_match: float = 0.25):
        self.tracks: dict[int, Track] = {}
        self.next_id = 1
        self.max_missed = max_missed
        self.iou_match = iou_match

    @staticmethod
    def _iou(a, b) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1e-6)

    def update(self, detections: list[Detection], frame_idx: int) -> list[Track]:
        matched_tracks = set()
        for det in detections:
            best, best_iou = None, self.iou_match
            for tid, tr in self.tracks.items():
                if tid in matched_tracks:
                    continue
                iou = self._iou(tr.bbox, det.bbox)
                if iou > best_iou:
                    best, best_iou = tid, iou
            if best is None:
                tr = Track(track_id=self.next_id, bbox=det.bbox)
                self.next_id += 1
                self.tracks[tr.track_id] = tr
                best = tr.track_id
            tr = self.tracks[best]
            matched_tracks.add(best)
            tr.bbox = det.bbox
            tr.frames_seen += 1
            tr.hardhat_votes.append(det.hardhat)
            tr.hardhat_votes = tr.hardhat_votes[-15:]
            tr.motion = det.motion
            if det.in_restricted_zone:
                tr.zone_frames += 1
            else:
                tr.zone_frames = 0
            if (not det.hardhat) or det.in_restricted_zone:
                tr.violation_frames += 1
            else:
                tr.violation_frames = 0
            tr.last_seen_frame = frame_idx
            det.track_id = best
        # Expire stale tracks.
        stale = [
            tid
            for tid, tr in self.tracks.items()
            if frame_idx - tr.last_seen_frame > self.max_missed
        ]
        for tid in stale:
            del self.tracks[tid]
        return [self.tracks[tid] for tid in matched_tracks]


class PerceptionPipeline:
    """One-call perception: detect -> classify PPE -> zone check -> track."""

    def __init__(self, cfg, zones):
        self.detector = PersonDetector(
            model_path=cfg.perception.model_path,
            conf_thresh=cfg.perception.dnn_conf_thresh,
            nms_thresh=cfg.perception.dnn_nms_thresh,
            input_size=cfg.perception.dnn_input_size,
        )
        self.hardhat = HardhatClassifier(cfg.perception)
        self.zones = ZoneChecker(zones)
        self.flow = MotionEstimator()
        self.tracker = CentroidTracker()
        self.cfg = cfg

    def process(self, frame: np.ndarray, frame_idx: int):
        motion = self.flow.estimate(frame)
        detections = self.detector.detect(frame)
        for det in detections:
            det.hardhat, det.hardhat_confidence = self.hardhat.classify(frame, det.bbox)
            det.in_restricted_zone, det.zone_name = self.zones.check(frame.shape, det.bbox)
            det.motion = motion
        tracks = self.tracker.update(detections, frame_idx)
        return detections, tracks, motion
