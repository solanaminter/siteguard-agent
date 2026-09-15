"""Perception unit tests — synthetic images, no model download needed."""
import cv2
import numpy as np

from siteguard.config import PerceptionConfig, Zone
from siteguard.perception import (
    CentroidTracker, Detection, HardhatClassifier, MotionEstimator, ZoneChecker,
)


def head_roi_with_helmet(color_bgr=(0, 215, 255)):
    """Fake 'person' crop: skin-tone lower head + colored helmet dome on top.

    The dome is centered and narrower than the head (like a real hardhat
    inside the classifier's inset head ROI).
    """
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    img[:, :] = (128, 128, 128)  # neutral gray background (no helmet hues)
    img[15:55, 30:70] = color_bgr  # helmet dome (compact, centered)
    img[60:120, 20:80] = (120, 150, 200)  # face-ish
    return img


def test_hardhat_detected_yellow():
    clf = HardhatClassifier(PerceptionConfig())
    img = head_roi_with_helmet((0, 215, 255))  # yellow in BGR
    ok, conf = clf.classify(img, (0, 0, 100, 200))
    assert ok and conf > 0.5


def test_hardhat_detected_white():
    clf = HardhatClassifier(PerceptionConfig())
    img = head_roi_with_helmet((240, 240, 240))
    ok, conf = clf.classify(img, (0, 0, 100, 200))
    assert ok


def test_no_hardhat_bare_head():
    clf = HardhatClassifier(PerceptionConfig())
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    img[:, :] = (128, 128, 128)  # neutral gray background (no helmet hues)
    img[0:120, 20:80] = (40, 50, 80)  # dark hair, no helmet colors
    ok, conf = clf.classify(img, (0, 0, 100, 200))
    assert not ok


def test_zone_checker_inside_outside():
    zones = ZoneChecker([Zone(name="z", polygon=[(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)])])
    inside, name = zones.check((480, 640, 3), (500, 350, 80, 100))
    assert inside and name == "z"
    outside, _ = zones.check((480, 640, 3), (10, 10, 80, 100))
    assert not outside


def test_tracker_assigns_stable_ids():
    tr = CentroidTracker()
    d1 = [Detection(bbox=(10, 10, 50, 100))]
    tracks = tr.update(d1, 0)
    tid = tracks[0].track_id
    d2 = [Detection(bbox=(12, 12, 50, 100))]
    tracks = tr.update(d2, 1)
    assert tracks[0].track_id == tid
    assert tracks[0].frames_seen == 2


def test_tracker_expires_stale():
    tr = CentroidTracker(max_missed=2)
    tr.update([Detection(bbox=(10, 10, 50, 100))], 0)
    tr.update([], 1)
    tr.update([], 2)
    remaining = tr.update([], 5)
    assert remaining == [] and tr.tracks == {}


def test_motion_estimator_zero_on_identical():
    me = MotionEstimator()
    img = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
    me.estimate(img)
    assert me.estimate(img) < 0.5


def test_motion_estimator_detects_shift():
    me = MotionEstimator()
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (60, 60), (255, 255, 255), -1)
    me.estimate(img)
    img2 = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(img2, (40, 20), (80, 60), (255, 255, 255), -1)
    assert me.estimate(img2) > 1.0
