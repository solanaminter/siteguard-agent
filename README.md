# SiteGuard Agent

An **agentic vision system** for construction-site safety: OpenCV 5 perception
drives a risk-reasoning agent loop where **vision output changes downstream
decisions** — detections raise incidents, persistence escalates them, cleared
PPE withdraws them, and a human safety officer approves or rejects actions.

Built for the **OpenCV AI Competition 2026** (targeting the overall ranking +
the **Agentic Vision Award**). Solo build by **Dy (Solana Minter) / Green Robin**.

## Live Demo

> 🌐 **Deployed demo site:** https://siteguard-agent-demo.vercel.app
>
> Demo video: https://www.youtube.com/watch?v=FMDSdMdh9Oc
> Option A: `cd web && vercel deploy` (public demo site).
> Option B: `infra/deploy.sh <vpc-id> <subnet-ids>` (AWS live endpoint).
> Paste the resulting URL here and into the Devpost "Try it out" field.

## Demo video

**[`assets/siteguard-demo.mp4`](assets/siteguard-demo.mp4)** (3:58) — problem, real end-to-end pipeline,
trace-proven decision changes, architecture, honest metrics and failure cases,
responsible use.

> 🎬 YouTube (public/unlisted, judge-accessible): **TODO — parent uploads
> `assets/siteguard-demo.mp4` and pastes the link here.**
>
> Suggested title: `SiteGuard Agent — Agentic Construction-Site Safety with
> OpenCV 5 | OpenCV AI Competition 2026`

## The loop

```
site camera frames
  -> OpenCV 5 perception: cv2.dnn person detection (YOLOX-s ONNX from the
     official OpenCV Model Zoo),
     HSV hardhat segmentation, polygon zone checks, Farneback optical flow,
     IoU centroid tracking
  -> risk scoring: no_hardhat +40, restricted-zone +35, persistence +10,
     motion +5, repeat offender +15
  -> policy:  <40 dismiss (log only)
              40-79 request HUMAN APPROVAL (review console)
              >=80  auto-escalate (evidence clip + safety alert)
  -> human-in-the-loop: approve / reject in the web console
  -> every step appended to a JSON trace
```

Vision changing decisions (the agentic part), all visible in `trace.json`:
- worker appears without hardhat -> `request_approval`
- worker puts hardhat on -> `request_approval -> withdrawn` ("vision: hardhat detected")
- violation persists / enters restricted zone -> `request_approval -> escalate`
- officer approves -> `safety_alert` action executes with evidence

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) fetch the pinned detection model (verifies SHA256)
bash scripts/download_models.sh        # -> assets/models/yolox_s_2022nov.onnx

# 2) run the demo service
uvicorn siteguard.app:app --port 8000  # open http://localhost:8000

# 3) run the staged end-to-end scenario (builds demo_input.mp4 first)
.venv/bin/python scripts/make_demo_clips.py
.venv/bin/python scripts/process_demo.py   # -> assets/annotated/demo_annotated.mp4
                                           # -> assets/annotated/trace.json

# 4) tests + eval
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m siteguard.eval       # -> docs/eval_results.json
```

## Models

`assets/models/yolox_s_2022nov.onnx` — YOLOX-s from the official OpenCV Model
Zoo (Apache 2.0, tuned for the OpenCV `dnn` module), run with
`cv2.dnn.readNetFromONNX` (OpenCV 5 dropped the Darknet/HOG importers present
in 4.x). Pinned SHA256 in `assets/models/SHA256SUMS`;
`scripts/download_models.sh` fetches + verifies. The model file itself is
git-ignored; the Docker build re-fetches it.

Hardhat classification is pure classical CV (no learned model): HSV
segmentation of the head ROI for helmet colors + morphological cleanup,
with three OR-paths (dome-shaped blob, yellow dominance, white + edge
texture) so close-up helmets that fill the ROI are still caught.

## Web API

| Endpoint | Description |
|---|---|
| `GET /` | review console (incidents, approve/reject, trace) |
| `GET /health` | liveness probe (ALB target-group health check) |
| `POST /api/analyze/image` | single-frame perception + agent step |
| `POST /api/analyze/video` | video -> incidents + annotated video |
| `GET /api/incidents` | incident list |
| `POST /api/incidents/{id}/approve` | human approval -> action executes |
| `POST /api/incidents/{id}/reject` | human rejection |
| `GET /api/trace` | full agent trace (`?event=decision_change`) |

## AWS deployment

Containerized (see `infra/`): ECR image -> ECS Fargate service, S3 evidence
bucket, CloudWatch structured-JSON logs. One command (needs AWS creds):

```bash
AWS_PROFILE=<you> ./infra/deploy.sh <vpc-id> <subnet-id-1,subnet-id-2>
```

Local parity: `STORAGE_BACKEND=local|s3` switches evidence storage; the Docker
Compose file runs the same image locally.

## Evaluation

`python -m siteguard.eval` runs the pipeline over the hand-labeled set
(`assets/raw/labels.json`, 8 scenes incl. failure probes: white helmet vs bright
sky, occlusion, dusk low light, empty scene, 11-worker crowd) and reports
detection precision/recall, hardhat classification accuracy, zone accuracy, and
latency. Results: `docs/eval_results.json`. Failure cases are documented in
`TECHNICAL_REPORT.md`.

## Layout

```
siteguard/          perception, agent loop, FastAPI app, storage, eval
scripts/            demo clip synthesis, scenario runner, arch diagram
infra/              Dockerfile, docker-compose, CloudFormation, deploy.sh
assets/raw/         synthetic demo media + labels.json (AI-generated stills)
assets/annotated/   demo_input.mp4, demo_annotated.mp4, trace.json
assets/models/      pinned model + SHA256SUMS (weights git-ignored)
docs/               eval results, demo summary
ARCHITECTURE.png    system diagram
TECHNICAL_REPORT.md competition technical report
SUBMISSION_BRIEF.md submission checklist + proposal draft
```

## Responsible use

This is a **decision-support** tool, not an autonomous enforcer: medium-risk
detections always wait for a human; only high-severity, persistent violations
auto-escalate, and every action carries evidence + a trace. See
`TECHNICAL_REPORT.md` ("Responsible use") for bias, privacy, and failure-mode
notes.
