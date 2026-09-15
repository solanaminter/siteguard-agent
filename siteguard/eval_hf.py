"""Eval harness on the real-world HF hard-hat subset
(eval-data/hf-hardhat): head-level COCO labels over genuine
construction-site photos.

The source labels HEADS (hardhat / no-hardhat), not persons, so:
  - a GT head is "covered" when its center falls inside a detected
    person box (head-coverage recall);
  - classification metrics compare the matched person's hardhat
    prediction against the GT head label;
  - detection precision is NOT reported here: a person detection with
    no labeled head may still be a correct person the annotators did
    not label.

Run:  python -m siteguard.eval_hf [--data-dir eval-data/hf-hardhat]
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from .config import AppConfig
from .perception import PerceptionPipeline

DATA_DIR = "eval-data/hf-hardhat"


def head_center_in_box(cx, cy, bbox) -> bool:
    x, y, w, h = bbox
    return x <= cx <= x + w and y <= cy <= y + h


def run_eval(data_dir=DATA_DIR):
    with open(os.path.join(data_dir, "annotations.json")) as f:
        coco = json.load(f)
    img_dir = os.path.join(data_dir, "images")
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    img_by_id = {i["id"]: i for i in coco["images"]}

    cfg = AppConfig()
    hat_tp = hat_fp = hat_tn = hat_fn = 0
    covered = total_heads = 0
    unmatched_dets = 0
    latencies = []
    per_image = []

    ids = sorted(by_img)
    for n, iid in enumerate(ids):
        meta = img_by_id[iid]
        path = os.path.join(img_dir, os.path.basename(meta["file_name"]))
        img = cv2.imread(path)
        assert img is not None, f"missing {path}"
        pipe = PerceptionPipeline(cfg, cfg.zones)  # fresh: no cross-image state
        t0 = time.perf_counter()
        detections, _, _ = pipe.process(img, 0)
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

        anns = by_img[iid]
        matched_det = set()
        img_cov = 0
        for a in anns:
            x, y, w, h = a["bbox"]
            cx, cy = x + w / 2.0, y + h / 2.0
            total_heads += 1
            best, best_conf = -1, -1.0
            for di, d in enumerate(detections):
                if head_center_in_box(cx, cy, d.bbox) and d.confidence > best_conf:
                    best, best_conf = di, d.confidence
            if best < 0:
                continue
            covered += 1
            img_cov += 1
            matched_det.add(best)
            gt_hat = a["category_id"] == 0
            pred_hat = detections[best].hardhat
            if pred_hat and gt_hat:
                hat_tp += 1
            elif pred_hat and not gt_hat:
                hat_fp += 1
            elif not pred_hat and not gt_hat:
                hat_tn += 1
            else:
                hat_fn += 1
        unmatched_dets += len(detections) - len(matched_det)
        per_image.append({
            "image": os.path.basename(meta["file_name"]),
            "latency_ms": round(dt, 1),
            "detections": len(detections),
            "gt_heads": len(anns),
            "heads_covered": img_cov,
        })
        if (n + 1) % 50 == 0:
            print(f"  ... {n + 1}/{len(ids)} images", flush=True)

    def pr(tp, fp, fn):
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        return round(p, 3), round(r, 3), round(f1, 3)

    hat_p, hat_r, hat_f1 = pr(hat_tp, hat_fp, hat_fn)
    hat_acc = (hat_tp + hat_tn) / max(hat_tp + hat_fp + hat_tn + hat_fn, 1)
    summary = {
        "dataset": ("keremberke/hard-hat-detection valid subset "
                    "(CC BY 4.0, 200 images, seed 20260915)"),
        "hardhat_classification": {
            "precision": hat_p, "recall": hat_r, "f1": hat_f1,
            "accuracy": round(hat_acc, 3),
            "tp": hat_tp, "fp": hat_fp, "tn": hat_tn, "fn": hat_fn,
        },
        "head_coverage_recall": round(covered / max(total_heads, 1), 3),
        "heads_covered": covered,
        "heads_total": total_heads,
        "detections_without_labeled_head": unmatched_dets,
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 1),
            "p95": round(float(np.percentile(latencies, 95)), 1),
            "backend": "cv2.dnn CPU (OpenCV 5.0.0)",
        },
        "n_images": len(ids),
    }
    os.makedirs("docs", exist_ok=True)
    with open("docs/eval_hf_results.json", "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    args = ap.parse_args()
    run_eval(args.data_dir)


if __name__ == "__main__":
    main()
