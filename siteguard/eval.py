"""Eval harness: precision/recall for person detection, hardhat
classification accuracy, zone-intrusion accuracy, and per-frame latency
over the hand-labeled set in assets/raw/labels.json.

Label schema per image:
  {"image": "<file>", "zones": {<zone-name>: <polygon rel-coords>} | null,
   "persons": [{"bbox": [x,y,w,h] (relative 0..1), "hardhat": bool,
                "in_zone": bool, "note": str}], "notes": str}

Run:  python -m siteguard.eval
"""
import glob
import json
import os
import time

import cv2
import numpy as np

from .config import AppConfig, Zone
from .perception import PerceptionPipeline

RAW = "assets/raw"
LABELS = os.path.join(RAW, "labels.json")
IOU_MATCH = 0.5


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-9)


def run_eval():
    with open(LABELS) as f:
        labels = json.load(f)
    cfg = AppConfig()
    results = {
        "det_tp": 0, "det_fp": 0, "det_fn": 0,
        "hat_tp": 0, "hat_fp": 0, "hat_tn": 0, "hat_fn": 0,
        "zone_correct": 0, "zone_total": 0,
        "latencies_ms": [],
        "per_image": [],
    }
    for item in labels:
        path = os.path.join(RAW, item["image"])
        img = cv2.imread(path)
        assert img is not None, f"missing {path}"
        h, w = img.shape[:2]
        zone_cfg = item.get("zones")
        if zone_cfg is None:
            zones = cfg.zones
        else:
            zones = [Zone(name=n, polygon=p, restricted=True)
                     for n, p in zone_cfg.items()]
        pipe = PerceptionPipeline(cfg, zones)
        t0 = time.perf_counter()
        detections, tracks, motion = pipe.process(img, 0)
        dt = (time.perf_counter() - t0) * 1000
        results["latencies_ms"].append(dt)

        gt_boxes = [
            (int(p["bbox"][0] * w), int(p["bbox"][1] * h),
             int(p["bbox"][2] * w), int(p["bbox"][3] * h))
            for p in item["persons"]
        ]
        matched_gt, matched_det = set(), set()
        pairs = []
        for di, d in enumerate(detections):
            best, best_iou = -1, IOU_MATCH
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                v = iou(d.bbox, gb)
                if v > best_iou:
                    best, best_iou = gi, v
            if best >= 0:
                matched_gt.add(best)
                matched_det.add(di)
                pairs.append((di, best))
        results["det_tp"] += len(pairs)
        results["det_fp"] += len(detections) - len(pairs)
        results["det_fn"] += len(gt_boxes) - len(pairs)

        hat_ok = zone_ok = 0
        for di, gi in pairs:
            gt = item["persons"][gi]
            pred_hat = detections[di].hardhat
            if pred_hat and gt["hardhat"]:
                results["hat_tp"] += 1
            elif pred_hat and not gt["hardhat"]:
                results["hat_fp"] += 1
            elif not pred_hat and not gt["hardhat"]:
                results["hat_tn"] += 1
            else:
                results["hat_fn"] += 1
            if detections[di].in_restricted_zone == gt["in_zone"]:
                results["zone_correct"] += 1
            results["zone_total"] += 1
        results["per_image"].append({
            "image": item["image"],
            "latency_ms": round(dt, 1),
            "detections": len(detections),
            "gt_persons": len(gt_boxes),
            "matched": len(pairs),
            "notes": item.get("notes", ""),
        })

    def pr(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return round(p, 3), round(r, 3), round(f1, 3)

    det_p, det_r, det_f1 = pr(results["det_tp"], results["det_fp"], results["det_fn"])
    hat_p, hat_r, hat_f1 = pr(results["hat_tp"], results["hat_fp"], results["hat_fn"])
    hat_acc = ((results["hat_tp"] + results["hat_tn"])
               / max(results["hat_tp"] + results["hat_fp"]
                     + results["hat_tn"] + results["hat_fn"], 1))
    summary = {
        "detection": {"precision": det_p, "recall": det_r, "f1": det_f1,
                      "tp": results["det_tp"], "fp": results["det_fp"],
                      "fn": results["det_fn"]},
        "hardhat_classification": {"precision": hat_p, "recall": hat_r,
                                   "f1": hat_f1, "accuracy": round(hat_acc, 3),
                                   "tp": results["hat_tp"], "fp": results["hat_fp"],
                                   "tn": results["hat_tn"], "fn": results["hat_fn"]},
        "zone_accuracy": round(results["zone_correct"] / max(results["zone_total"], 1), 3),
        "latency_ms": {
            "mean": round(float(np.mean(results["latencies_ms"])), 1),
            "p95": round(float(np.percentile(results["latencies_ms"], 95)), 1),
        },
        "n_images": len(labels),
    }
    os.makedirs("docs", exist_ok=True)
    with open("docs/eval_results.json", "w") as f:
        json.dump({"summary": summary, "per_image": results["per_image"]}, f, indent=2)
    print(json.dumps(summary, indent=2))
    for pi in results["per_image"]:
        print(f"  {pi['image'][:42]:42s} det={pi['detections']}/{pi['gt_persons']} "
              f"matched={pi['matched']} {pi['latency_ms']:.0f}ms  {pi['notes'][:40]}")
    return summary


if __name__ == "__main__":
    run_eval()
