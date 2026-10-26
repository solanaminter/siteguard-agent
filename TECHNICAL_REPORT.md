# SiteGuard Agent — Technical Report
### OpenCV AI Competition 2026 · Solo entry by Dy (Solana Minter) / Purple Castle Ventures

## 1. Problem and users

Construction sites kill or injure workers every day, and a large share of
incidents trace back to two preventable causes: missing personal protective
equipment (PPE) and workers entering restricted zones (excavations, crane
swing radii, live electrical areas). Safety officers cannot watch every
camera feed at once, and fixed rule-based alarms ("person in polygon →
siren") generate so many false positives that crews learn to ignore them.

**SiteGuard Agent** is an agentic vision system for construction-site safety
monitoring. It watches site camera feeds with OpenCV 5 perception, reasons
about risk per tracked worker, and acts through a tiered policy:

- **low risk** → dismiss (log only),
- **medium risk** → request **human approval** (a safety officer reviews),
- **high risk** → **auto-escalate** (save evidence, dispatch a safety alert).

The key agentic property: **vision output changes downstream decisions**.
A pending approval is not a fixed verdict — if later frames show the worker
putting on a hardhat, the request is *withdrawn*; if the violation persists
or the worker enters a restricted zone, it is *escalated*. Every step is
appended to a JSON trace, so any decision can be audited frame by frame.

Primary users: site safety officers (review console, approve/reject),
site managers (incident history, trace audits), and safety-system
integrators (REST API, AWS deployment).

## 2. System overview

```
site camera frames
  -> OpenCV 5 perception: cv2.dnn person detection (YOLOX-s ONNX, OpenCV Zoo)
     HSV hardhat segmentation, polygon zone checks, Farneback optical flow,
     IoU centroid tracking
  -> risk scoring: no_hardhat +40, restricted-zone +35, persistence +10,
     motion +5, repeat offender +15
  -> policy: <40 dismiss | 40-79 request human approval | >=80 auto-escalate
  -> human-in-the-loop: approve / reject in the review console
  -> JSON trace of every perception, decision, and action
```

See `ARCHITECTURE.png` for the full diagram (ingest → perception →
agent loop → AWS delivery).

## 3. OpenCV 5 perception pipeline

All computer vision is OpenCV 5 — no learned models beyond the person
detector, no cloud vision APIs.

**Person detection (`cv2.dnn`).** YOLOX-s from the official OpenCV Model Zoo
(`opencv/object_detection_yolox`, Apache 2.0), run via
`cv2.dnn.readNetFromONNX`. Preprocessing follows the Zoo reference:
BGR→RGB, 640×640 top-left letterbox with 114.0 fill, NCHW float32, no
normalization; postprocessing multiplies objectness × class score with NMS.
OpenCV 5 dropped the Darknet/HOG importers present in 4.x, so ONNX is the
supported path. Model pinned by SHA256 (`assets/models/SHA256SUMS`).

**Hardhat classification (`imgproc`, classical only).** The head ROI (top
15% of the person box, 15% side insets) is segmented in HSV for saturated
helmet colors (yellow/orange/red/blue) and white, cleaned with
`morphologyEx(OPEN)`, then decided by three OR-paths:

1. **Dome shape** — a compact, solid (`convexHull` solidity ≥ 0.82),
   dome-proportioned (`fitEllipse` aspect 1.1–3.5) blob that does not span
   the ROI width (sky, dirt mounds, and walls do);
2. **Yellow dominance** — yellow-hue coverage ≥ 10% (catches close-up
   yellow hardhats that fill the ROI; measured dirt/sky confounders at
   ~0.00 vs. 0.07–0.47 for real yellow hats);
3. **White + texture** — a dome-like white blob with brim/ridge edge
   texture (`Canny` edge density > 0.015; smooth sky has none).

Bare heads fail all three: background blobs span the ROI or lack helmet
hues. Yellow starts at H=20 because brown dirt — the main confounder —
peaks below H=20 while yellow hardhats peak at 20–35.

**Restricted zones (`imgproc`).** `cv2.pointPolygonTest` on the detection
center against configured polygons (e.g. the taped excavation area).

**Motion (`video`).** Dense Farneback optical flow; the 95th-percentile
magnitude flags activity. The estimator resets when frame resolution
changes so one pipeline can be reused across stills.

**Tracking.** Lightweight IoU centroid tracker (match ≥ 0.25, 10-frame
grace) with a 15-frame hardhat majority vote per track, which smooths
single-frame classifier flicker and is what lets a *sustained* vision
signal (worker puts hat on) change an *open* decision.

## 4. Agentic decision loop

`SafetyAgent.observe()` runs once per frame over active tracks:

- `risk_score()` sums weighted evidence; `_decide()` maps to
  dismiss / request_approval / escalate.
- A medium-risk track opens an **incident** (`INC-0001`…) and logs
  `approval_requested`. The incident stays **pending** while frames stream.
- **Later vision changes the decision:**
  - hardhat majority vote flips to compliant → risk drops below 40 →
    `request_approval → withdrawn` ("vision: hardhat detected");
  - violation persists 45+ frames or zone evidence appears → risk crosses
    80 → `request_approval → escalate` ("vision: persistent violation /
    zone intrusion"), executing the safety alert automatically;
  - the track vanishes → `request_approval → dismissed` ("track lost").
- A safety officer can `approve()` / `reject()` any pending incident from
  the review console or API; approval executes the alert with the evidence
  frame attached. Rejection marks the track a repeat offender (+15 if it
  re-offends).

Every perception, incident opening, approval request, decision change,
human decision, and action is appended to a timestamped JSON trace
(`assets/annotated/trace.json` in the demo), making the "vision changed the
decision" moments explicitly auditable — the core of the Agentic Vision
Award criterion.

## 5. Demo scenario (real pipeline, staged footage)

`scripts/make_demo_clips.py` synthesizes a 27-second, 5-segment camera feed
from AI-generated stills with slow Ken Burns pans (the pan of the
"corrected" segment starts where the "violation" pan ended, so the IoU
tracker holds one track across the cut); `scripts/process_demo.py` runs the
actual perception + agent loop over all 405 frames:

| Segment | Content | Agent behavior |
|---|---|---|
| CAM 01 (0–5 s) | two workers, hardhats on | dismiss (log only) |
| CAM 02 (5–10 s) | worker, bare head | `request_approval` → **pending** (INC-0001, f83) |
| CAM 02 (10–15 s) | *same worker puts hardhat on* | `pending → **withdrawn**` at f157 (vision changed the decision) |
| CAM 03 (15–22 s) | worker, no hardhat, inside excavation zone | `pending → **escalated**` at f228 (INC-0003, risk 90; auto safety alert + evidence) |
| CAM 04 (22–27 s) | worker, bare head | `request_approval` → officer **approves** (INC-0004) → alert executes |

Outputs: `assets/annotated/demo_annotated.mp4` (annotated footage),
`assets/annotated/trace.json` (full trace), `docs/demo_summary.json`.

## 6. Evaluation

`python -m siteguard.eval` runs the pipeline over 8 hand-labeled scenes
(`assets/raw/labels.json`; AI-generated stills, boxes verified against
detector output, conservative labeling — only clearly-identifiable workers
labeled). Results (`docs/eval_results.json`):

| Metric | Result |
|---|---|
| Detection precision / recall / F1 | **0.905 / 0.905 / 0.905** (19 TP, 2 FP, 2 FN) |
| Hardhat precision / recall / F1 / accuracy | **1.00 / 0.875 / 0.933 / 0.895** (14 TP, 0 FP, 3 TN, 2 FN) |
| Zone accuracy | **1.00** |
| Latency (CPU) | mean **953 ms**, p95 **1114 ms** per frame |

**Failure cases (kept as explicit probes, not hidden):**
- `whitehat_sky.webp` — white hardhat against overexposed sky merges into
  the background blob and is missed; the hanging safety netting also draws
  a false person detection. Documents the known white-on-white limitation.
- `occlusion.webp` — the 4th worker, heavily occluded behind scaffolding
  poles, is missed; the yellow excavator arm draws a false person
  detection (a real deployment would mask known equipment or use an
  equipment class).
- `overview.webp` — a distant worker on the building top and a distant
  blue hardhat are missed (small-object recall limits of a 640-px
  detector on CPU).
- `dusk.webp` — low-light detection works (1/1) but the bare-head worker
  is correctly flagged; classifier margins shrink in dim light.

**Limitations.** CPU inference (~1 s/frame on the synthetic set, ~0.48 s/frame on the
real-world set) is not real-time 15 fps; a
production deployment would need a GPU-enabled container build or
frame-skipping (the current image is CPU-only). The 8-scene set is small
and synthetic — it measures the pipeline honestly but is not a
substitute for on-site validation. Single-camera tracking; multi-camera
re-identification is out of scope.

**Real-world hardhat eval (`python -m siteguard.eval_hf`, run 2026-09-15, CPU).**
200 images from `keremberke/hard-hat-detection` (valid split, CC BY 4.0,
deterministic seed 20260915), genuine construction-site photos labeled at
the head level. Source labels heads, not persons, so metrics use
head-coverage matching. Results (`docs/eval_hf_results.json`):

| Metric | Result |
|---|---|
| Hardhat classification precision / recall / F1 / accuracy | **0.795 / 0.748 / 0.771 / 0.746** (175 TP, 45 FP, 130 TN, 59 FN) |
| Head-coverage recall (GT head centers inside a person box) | **0.812** (409/504) |
| Latency (CPU) | mean **478 ms**, p95 **548 ms** per image |

As expected, real-world accuracy is below the synthetic 8-scene set —
real lighting, hardhat variety, and occlusions are harder. This is the
honest number to quote to judges; it motivates the human-in-the-loop
design.

## 7. AWS architecture

Containerized delivery (`infra/`), designed for a one-command deploy by
the operator (`infra/deploy.sh <vpc-id> <subnet-ids>`):

- **ECR → ECS Fargate** service (1 task, 1 vCPU / 2 GB) running
  `uvicorn siteguard.app:app` behind an **Application Load Balancer**
  (public HTTP listener → target group with `/health` checks; tasks only
  accept traffic from the ALB security group).
- **S3** evidence bucket (blocked public access, versioning, 90-day
  lifecycle) for incident evidence frames; `STORAGE_BACKEND=local|s3`
  keeps local dev parity.
- **CloudWatch** log group (`/ecs/siteguard-agent`) receiving the
  structured JSON logs (one line = one event), suitable for metric
  filters/alarms on escalation rate.
- **CloudFormation** template (`infra/cloudformation.yaml`) declares all
  of the above. Live AWS deployment verified Sep 15, 2026 (ECS Fargate +
  ALB endpoint returned HTTP 200; full evidence in
  `AWS_DEPLOY_EVIDENCE.md`); the stack was torn down after verification
  to stop billing, and the judge-facing live endpoint is the Vercel demo
  (see README). Redeploy is one command (`infra/deploy.sh`) if a live AWS
  endpoint is wanted again.

## 8. Responsible use

SiteGuard is a **decision-support** tool, not an autonomous enforcer.
Medium-risk detections always wait for a human; only high-severity,
persistent violations auto-escalate, and every action carries evidence plus
a full trace. Notes for deployment:

- **Bias.** The eval imagery is AI-generated and may not represent the
  full diversity of workers, skin tones, helmet colors, or site
  conditions. Validate on real site footage before trusting the
  classifier, and monitor false-positive rates per site.
- **Privacy.** The system detects *persons* and PPE — it performs no
  facial recognition and no identity tracking. Evidence clips should be
  retained per site policy (default 90 days) and access-controlled.
- **Failure modes.** Documented above; the trace makes every miss or false
  alarm auditable. The white-on-white and heavy-occlusion misses are
  reasons to keep the human in the loop, not to remove them.
- **No autonomous punishment.** Alerts go to safety staff for response;
  the system must not be wired to turnstiles, payroll, or disciplinary
  actions without human review.

## 9. Reproducibility

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh        # pinned YOLOX-s, SHA256-verified
.venv/bin/python -m pytest tests/ -q    # 17 unit tests
.venv/bin/python -m siteguard.eval     # metrics -> docs/eval_results.json
.venv/bin/python scripts/make_demo_clips.py
.venv/bin/python scripts/process_demo.py  # demo -> assets/annotated/
```

Model: `assets/models/yolox_s_2022nov.onnx` (OpenCV Zoo, Apache 2.0),
SHA256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.

---
*Built by Dy (Solana Minter) / Purple Castle Ventures for the OpenCV AI Competition 2026.*
*Status update 2026-10-25: filed to the OpenCV AI Competition 2026 on
2026-10-23 (editable until Oct 26, 11:45 PM PT); demo and video live
(see README). The "local build only" line from earlier drafts is superseded
by that filing — see SUBMISSION_BRIEF.md for the current filing status.*
