# SUBMISSION BRIEF — SiteGuard Agent · OpenCV AI Competition 2026

> ## ⚠️ PROPOSAL URGENCY — READ FIRST
>
> The competition's build phase (Aug 26 – Oct 26) was open to "all teams that
> submitted the required proposal." The proposal window ran **August 12–26, 2026**
> via Jotform ("Apply for your grant here"). **Proposal eligibility for this
> entry is UNCERTAIN — it is not known whether a late proposal will be
> accepted.** If Dy wants this entry to be eligible, **submit the Jotform
> proposal IMMEDIATELY** using the paste-ready text below. Do not wait for the
> video upload or final polish. The proposal is the gating item; everything
> else can follow.
>
> **STATUS UPDATE 2026-10-22 (supersedes the "nothing submitted" note below
> where it conflicts):** Jotform proposal WAS submitted Sep 14, 2026 ~21:05 PDT
> (confirmation received); AWS deployment evidence completed Sep 15
> (live endpoint verified, stack torn down after verification); demo video is
> live on YouTube (verified 2026-10-22). STILL OPEN ahead of the Oct 26 close:
> Devpost submission itself, and whether the late proposal was accepted —
> acceptance still unknown. Do not treat the proposal as accepted.
>
> **STATUS UPDATE 2026-10-23 ~08:05 PT (re-verified on opencv26.devpost.com):
> proposal-acceptance is NOT a gate for the final project.** The official page
> states the Jotform proposals were for the $150 cloud compute grant only, and
> "All other teams, including those that submit the cloud compute grant
> proposal remain eligible to continue to the two-month build phase." Final
> projects are due Oct 26, 2026 at 11:59 p.m. PT (page event window ends
> 11:45 p.m. PT — file by the earlier time). Final requirements checklist vs.
> this package: technical report ✓ (TECHNICAL_REPORT.md), public repo ✓
> (https://github.com/solanaminter/siteguard-agent, public 200 — NOTE: the
> correct repo name is siteguard-agent, not siteguard), pinned deps +
> instructions (verify at filing), architecture diagram ✓ (ARCHITECTURE.png),
> working web endpoint ✓ (https://siteguard-agent-demo.vercel.app, 200
> 2026-10-23), video ≤5 min ✓ (3:58, public via oEmbed 2026-10-23,
> https://www.youtube.com/watch?v=FMDSdMdh9Oc — but CHECK at filing whether
> the video shows the team, as the rules ask the video to show "the team, the
> application working, its architecture, and its principal results"),
> evaluation evidence incl. failure cases ✓ (docs/eval_results.json). REMAINING
> before filing: staffed-window push (local main is 5 commits ahead of origin,
> incl. the Purple Castle Ventures affiliation fix; a non-interactive push
> fails without GitHub credentials in this run) + the Devpost submission
> itself. AWS note: the meaningful-AWS-component evidence is the Sep 15 live
> ECS/Fargate deployment (AWS_DEPLOY_EVIDENCE.md, stack torn down after
> verification); the live endpoint at filing time is the Vercel demo, which
> satisfies the "working web endpoint" alternative — do NOT claim the AWS
> stack is still live.
>
> **RE-CHECK 2026-10-23 ~08:55 PT (hourly run):** demo 200, repo
> https://github.com/solanaminter/siteguard-agent 200, video public via
> oEmbed — filing package still green; only the staffed-window push +
> Devpost submission remain.
>
> **AFFILIATION RESOLVED 2026-10-22 23:5x PT run:** filing affiliation is
> Purple Castle Ventures — Dy's hackathon company on file, used on every
> other campaign entry (employer Adobe is separate and is not the filing
> affiliation). §1 below now reads Purple Castle Ventures. The "Green
> Robin" lines remaining in §12 are the as-filed Sep 14 Jotform text
> (historical record — do NOT re-paste or edit); the Green Robin wording
> baked into the already-published YouTube video end card/description
> (§10) cannot be changed without a re-upload and is not filing copy.

---

## 1. Team

- **Team name:** SiteGuard Agent (solo)
- **Representative / sole member:** Solana Minter ("Dy") — solanaminter@gmail.com
- **Affiliation (filing):** Purple Castle Ventures (Dy is Founder; employer Adobe is not the filing affiliation — Green Robin is superseded since Sep 19)
- LinkedIn: https://linkedin.com/in/solanaminter
- GitHub: https://github.com/solanaminter
- Dy is **not** a student. Never claim student status.

## 2. Paths / categories

- **Primary target: Agentic Vision Award ($1,000)** — the entry is designed
  around it: a multi-step perception → decision → action loop in which OpenCV 5
  output changes later decisions (withdrawal of a pending approval; escalation
  of a pending request), with a full JSON trace as evidence.
- Also eligible for overall ranking (1st $5,000 / 2nd $3,000 / 3rd $2,000).
- **Not** pursuing the Best Use of COOL Award (no Graviton/Arm workload).

## 3. Problem & users

Construction sites kill and injure workers every year from preventable causes.
Two dominate: **missing hardhats** and **workers entering restricted zones**
(excavations, crane swing areas, live electrical). Safety officers cannot watch
every camera feed; fixed rule-based alarms cry wolf so often that crews ignore
them.

**Users:** site safety officers (review console: approve/reject pending
incidents), site managers (escalation alerts + evidence), and workers
(beneficiaries of faster intervention).

**Thesis:** an *agentic* vision monitor that reasons about risk, keeps a human
in the loop for medium risk, auto-escalates only persistent high-severity
violations — and, crucially, **revises open decisions when later vision
changes the picture** (hardhat appears → pending request withdrawn).

## 4. OpenCV 5 implementation (substantive)

- **Detection:** YOLOX-s from the official OpenCV Model Zoo, ONNX,
  `cv2.dnn.readNetFromONNX`, official preprocessing (BGR→RGB, 640×640
  letterbox, strides 8/16/32, objectness×class score, NMS).
- **Hardhat classification:** classical OpenCV 5 — HSV segmentation for
  yellow/orange/red/blue/white on the head ROI, three OR paths (compact
  saturated dome / yellow coverage ≥ 0.10 / white dome + Canny edge texture),
  contour area, convex-hull solidity, ellipse/aspect checks, morphology.
- **Zones:** `cv2.pointPolygonTest` against restricted-zone polygons.
- **Motion:** Farneback optical flow (`cv2.calcOpticalFlowFarneback`).
- **Tracking:** IoU centroid tracker with 15-frame hardhat majority vote.
- **Orchestration:** `imgproc` (segmentation), `dnn` (detection),
  `video` (motion) modules all exercised per frame.

## 5. Agentic loop (why it qualifies)

Risk policy: **<40 dismiss/log · 40–79 request human approval ·
≥80 auto-escalate.** A pending approval is not a fixed verdict:

- hardhat appears on the same track → `request_approval → withdrawn`
- violation persists / restricted-zone evidence → `request_approval → escalate`
- track vanishes → `request_approval → dismissed`
- officer can approve/reject any pending incident; approval executes the alert.

Every perception, incident, decision change, human decision, and action is
appended to a timestamped JSON trace (`assets/annotated/trace.json`).

**Demo proof (405 frames, `docs/demo_summary.json`):** 4 incidents, 640 trace
events —
INC-0001 `request_approval` (f83) → **withdrawn** (f157, "vision: hardhat
detected"); INC-0002 zone entry → withdrawn (f226); INC-0003
no_hardhat+restricted_zone risk 90 → **escalated** (f228, auto safety_alert);
INC-0004 → officer **approved** → alert executed.

## 6. AWS architecture (meaningful)

Docker → ECR → **ECS Fargate** behind an **internet-facing ALB**
(security group allows 8000 only from the ALB; target group health check on
`/health`), **S3** evidence bucket (versioned, 90-day lifecycle),
**CloudWatch** structured JSON log group (`/ecs/siteguard-agent`), all declared
in **CloudFormation** (12 resources), deployed by `infra/deploy.sh`
(one command, prints the `EndpointURL` output). FastAPI review console +
`/api/analyze/image`, `/api/incidents`, `/api/trace`.

**Status (supersedes the "NOT deployed" note): live AWS deployment verified
Sep 15 (ECS Fargate + ALB endpoint returned HTTP 200; evidence in
`AWS_DEPLOY_EVIDENCE.md`), stack torn down after verification to stop billing.
The judge-facing live endpoint is the Vercel demo
(https://siteguard-agent-demo.vercel.app, verified 200 on 2026-10-22);
redeploy is one command if a live AWS endpoint is wanted again.

## 7. Evaluation (exact, honest)

`python -m siteguard.eval` on 8 hand-labeled scenes (AI-generated stills,
conservative labeling, deliberate failure probes). `docs/eval_results.json`:

| Metric | Value |
|---|---|
| Person detection precision / recall / F1 | **0.905 / 0.905 / 0.905** (19 TP, 2 FP, 2 FN) |
| Hardhat classification precision / recall / F1 / accuracy | **1.000 / 0.875 / 0.933 / 0.895** |
| Restricted-zone accuracy | **1.000** |
| Mean latency / frame (CPU) | **952.5 ms** (p95 1114 ms) |

**Failure cases (documented, in video):** white hardhat vs. blown-out sky
(missed); worker buried behind scaffolding (missed) + yellow excavator arm
(false person); small/distant figures strain the 640px detector. This is a
small synthetic set — an honest pipeline check, not field validation.

**Real-world check** (`python -m siteguard.eval_hf`, run 2026-09-15, CPU):
200 genuine construction-site photos from `keremberke/hard-hat-detection`
(valid split, CC BY 4.0, seed 20260915). `docs/eval_hf_results.json`:
hardhat classification **P/R/F1/accuracy 0.795 / 0.748 / 0.771 / 0.746**
(175 TP, 45 FP, 130 TN, 59 FN); head-coverage recall **0.812** (409/504);
mean latency **478 ms** (p95 548 ms). Lower than the synthetic set, as
expected on real imagery — the honest number to quote, and the reason
medium risk always waits for a human.

## 8. Responsible use

Decision support, not an autonomous enforcer. Medium risk always waits for a
human. Only persistent high-severity violations auto-escalate, each with
evidence + audit trace. No facial recognition, no identity tracking — persons
and PPE only. Must never be wired to punishment without human review.

## 9. Repository contents (local)

`~/workspace/hackathons/opencv-2026/siteguard/` — FastAPI app, perception
package, agent/policy, eval, CloudFormation, Dockerfile, deploy script,
`ARCHITECTURE.png`, `TECHNICAL_REPORT.md`, `LICENSE` (MIT, © 2026
Solana Minter), `assets/annotated/demo_annotated.mp4` + `trace.json`,
`docs/demo_summary.json`, `docs/eval_results.json`, `videos/siteguard-demo/`
(HyperFrames source).

**STATUS UPDATE 2026-10-25 (supersedes "Not pushed anywhere" above):**
public repo https://github.com/solanaminter/siteguard-agent exists and is
public, but its remote HEAD is still the Sep 15 commit — local main is
ahead with the filing commits (Purple Castle Ventures affiliation, live
YouTube link, demo-page links, cover assets). Pushing local main before
the Oct 26, 11:45 PM PT close is the remaining repo-sync step; the
Devpost submission itself was filed Oct 23 and does not depend on that push.

## 10. Video

- Final: `assets/siteguard-demo.mp4` (≤5 min; current cut 3:58) — problem,
  real end-to-end pipeline, trace-proven decision changes, architecture,
  metrics, failure cases, responsible use, "built by Dy / Green Robin" card.
- **Cover image (generated 2026-10-22, 2048×1152 WebP, verified image
  bytes):** `assets/media-generation-siteguard-cover-0-e481f26e-a61d-4b28-8992-f4b83b00f5a2.webp`
  — upload as the Devpost thumbnail at filing (illustrative art, not an
  evidence metric).
- **YouTube upload DONE — video live and public, re-verified 2026-10-22
  via oEmbed:** https://www.youtube.com/watch?v=FMDSdMdh9Oc —
  "SiteGuard Agent -- Demo" by Solana Minter (local file 3:58, within the
  5-min cap). Link already in README; paste into Devpost at filing.
- YouTube title: `SiteGuard Agent — Agentic Construction-Site Safety with OpenCV 5 | OpenCV AI Competition 2026`
- YouTube description:
  > SiteGuard Agent is an agentic vision system for construction-site safety,
  > built with OpenCV 5 for the OpenCV AI Competition 2026 (powered by AWS).
  > OpenCV 5 perception (YOLOX-s detection, HSV hardhat segmentation, zone
  > polygons, optical flow, IoU tracking) feeds a risk-policy agent: dismiss /
  > request human approval / auto-escalate — and later vision revises open
  > decisions. Watch a pending approval get WITHDRAWN when a hardhat appears,
  > and a restricted-zone violation get ESCALATED automatically, all proven by
  > a timestamped JSON trace. Deployed on AWS (ECS Fargate + ALB, S3,
  > CloudWatch, CloudFormation). Built solo by Dy (Solana Minter), Green Robin.
  > Code + technical report: [GitHub link once pushed]

## 11. Exact Devpost answers (paste-ready)

**What it does / description:**
SiteGuard Agent is an agentic vision monitor for construction-site safety.
OpenCV 5 detects workers and analyzes PPE, zones, and motion; a risk-policy
agent dismisses, requests human approval, or auto-escalates — and revises open
decisions when later vision changes the picture. A timestamped JSON trace
records every perception, decision, and action.

**Try it out (live demo URL):** https://siteguard-agent-demo.vercel.app
(`cd web && vercel deploy`, or the `EndpointURL` from `infra/deploy.sh`).

**How it uses OpenCV 5:** YOLOX-s (official OpenCV Model Zoo, ONNX,
cv2.dnn) person detection; classical HSV hardhat segmentation with dome-shape
analysis; pointPolygonTest restricted zones; Farneback optical flow;
IoU tracking — orchestrated per frame.

**AWS:** Containerized FastAPI on ECS Fargate behind an ALB; S3 evidence
bucket; CloudWatch structured logs; full CloudFormation stack with
one-command deploy.

**Agentic Vision Award evidence:** agent workflow diagram
(`ARCHITECTURE.png`); trace (`assets/annotated/trace.json`) showing
`request_approval → withdrawn` (f157, hardhat appears) and
`request_approval → escalate` (f228, persistent zone violation); evaluation
with failure cases (`docs/eval_results.json`); human-in-the-loop
approve/reject console.

**Tagline (one line):** Agentic construction-site safety: OpenCV 5 vision
drives a risk-reasoning agent where later perception revises open decisions.

**Inspiration:**
Construction's two most preventable incident causes — missing hardhats and
restricted-zone intrusions — still injure workers because safety officers
can't watch every feed and fixed rule-based alarms cry wolf until crews
ignore them. The OpenCV AI Competition 2026's Agentic Vision path asks for
systems where perception changes downstream decisions; a safety monitor
that revises its own open decisions when later vision changes the picture
is the honest fit.

**Built With:** OpenCV 5 (cv2.dnn, imgproc, video modules) · YOLOX-s ONNX
(official OpenCV Model Zoo) · Python · FastAPI · Uvicorn · Docker · AWS
(ECS Fargate, ALB, ECR, S3, CloudWatch, CloudFormation) · NumPy · vanilla
HTML/CSS/JS demo site (Vercel-ready) · HyperFrames (demo video)

**Challenges we ran into:**
Classical HSV hardhat classification had to reject dirt mounds, walls, and
blown-out sky while catching yellow/white/orange/red/blue helmets — solved
with three OR paths (dome solidity, yellow dominance, white + Canny edge
texture). Keeping one tracker identity across a stitched Ken Burns cut
required matching the "corrected" pan start to the "violation" pan end so
IoU tracking held the track. OpenCV 5 dropped the old Darknet importers,
so ONNX via cv2.dnn was the supported detection path.

**Accomplishments that we're proud of:**
A real agentic loop proven by a timestamped JSON trace — a pending approval
gets WITHDRAWN when vision shows a hardhat appearing (f157) and a zone
violation gets auto-ESCALATED at risk 90 (f228); human-in-the-loop
approve/reject console with evidence; honest eval including failure probes
and a real-world HF hardhat subset (F1 0.771); one-command AWS deploy via
CloudFormation.

**What we learned:**
Classical CV + a detector can carry PPE compliance when the classifier is
built around the confounders (brown dirt peaks below H=20; sky spans the
ROI; smooth sky has no edge texture) rather than a single color threshold.
Synthetic scenes flatter metrics — the real-world subset (0.795 precision
vs. 1.000 synthetic) is why medium risk always waits for a human.

**What's next for SiteGuard Agent:**
Multi-camera re-identification, equipment-class masking (the yellow
excavator false positive), a GPU-enabled container build for real-time
throughput, and validation on real site footage before any field trust.

## 12. Jotform proposal — FILED Sep 14, 2026 ~21:05 PDT (historical record; do NOT re-file or edit the text below)

```
Team name: SiteGuard Agent (solo — Solana Minter, "Dy", Green Robin)

Problem statement & real-world impact:
Construction sites injure workers every year from preventable causes — chiefly
missing hardhats and workers entering restricted zones (excavations, crane
swing areas, live electrical). Safety officers cannot watch every camera, and
fixed alarms cry wolf until crews ignore them. SiteGuard Agent is an agentic
vision monitor that reasons about risk, keeps a human in the loop, and revises
its decisions when later vision changes the picture — cutting alarm fatigue
while catching the violations that matter.

Planned OpenCV 5 image/video analysis:
YOLOX-s (official OpenCV Model Zoo, ONNX, cv2.dnn) person detection at
1280×720/15fps; classical HSV hardhat segmentation with dome-shape analysis
on the head ROI; pointPolygonTest restricted-zone checks; Farneback optical
flow motion signal; IoU centroid tracking with hardhat majority vote —
orchestrated per frame through imgproc/dnn/video modules.

Planned AWS architecture & services:
Docker → ECR → ECS Fargate behind an Application Load Balancer; S3 evidence
bucket; CloudWatch structured JSON logs; everything declared in
CloudFormation and deployed with one script. FastAPI review console for
human approve/reject.

Architecture: see ARCHITECTURE.png (perception → agent loop → AWS delivery).

Target users: site safety officers (review console), site managers
(escalation alerts + evidence), construction workers (beneficiaries).

Evaluation method & judge demo:
8 hand-labeled scenes incl. failure probes (detection P/R, hardhat P/R,
zone accuracy, CPU latency); a scripted 27s multi-camera demo proving the
agentic loop — pending approval WITHDRAWN when a hardhat appears (trace
f157), restricted-zone violation auto-ESCALATED (trace f228) — with a
timestamped JSON trace as evidence. Judges get the video, the trace, and a
live endpoint or screen-share.

Paths: Agentic Vision path (primary target). Not pursuing COOL.

Team bio:
Solana Minter ("Dy") — solo builder at Green Robin. Ships end-to-end:
computer vision pipelines, agentic systems, and AWS delivery. Recent
hackathon submissions: Permit Pal (Agents for Humans Hackathon) and
Churn Radar (AI Builders Hackathon), each with repo + demo video.
```

## 13. Deployment steps (parent/Dy)

1. Submit Jotform proposal (Section 12) — **do this first**.
2. Push repo to GitHub (https://github.com/solanaminter).
3. Run `infra/deploy.sh` (needs AWS credentials) → note `EndpointURL`.
4. Upload `assets/siteguard-demo.mp4` to YouTube — DONE:
   https://www.youtube.com/watch?v=FMDSdMdh9Oc (verified public 2026-10-22).
5. Update README with GitHub + YouTube + endpoint links.
6. Fill Devpost submission (Section 11 answers + links + video + report).

## 14. Unresolved / unverified

- Proposal eligibility (late Jotform) — unknown; submitted ASAP per warning.
- AWS deployment — template validated, never deployed; costs unknown.
- Demo footage uses AI-generated stills (disclosed in report + video).
- Eval set is small/synthetic (disclosed); not field validation.

## 15. Statement of non-actions

**STATUS UPDATE 2026-10-25 (supersedes the builder-only note below where
it conflicts):** the Jotform proposal (Sep 14), AWS deployment verification
(Sep 15), YouTube upload, and Devpost submission (Oct 23) have all since
been performed; the only outward step still outstanding is pushing local
main to the public repo (see §9 update above).

No Jotform proposal, AWS deployment, GitHub push, YouTube upload, or Devpost
submission was performed by the builder. All artifacts are local-only under
`~/workspace/hackathons/opencv-2026/siteguard/`. Every outward step above is
for the parent agent or Dy to perform.

## Re-verification 2026-10-23 09:49 PT (hourly win loop)
- Demo https://siteguard-agent-demo.vercel.app 200; repo https://github.com/solanaminter/siteguard-agent 200; video public via oEmbed (https://www.youtube.com/watch?v=FMDSdMdh9Oc, 3:58, within 5-min cap).
- Filing (Oct 26, by 11:45 PM PT) still needs only: staffed-window push (local main 5 ahead of origin + 4 modified files + 1 untracked cover .json) + the Devpost submission itself. No push attempted this run (non-interactive push fails without GitHub credentials in this context, established prior runs).

## Re-verification 2026-10-23 11:55 PT (hourly win loop)
- Demo https://siteguard-agent-demo.vercel.app 200; video public via oEmbed (https://www.youtube.com/watch?v=FMDSdMdh9Oc).
- Filing (Oct 26, by 11:45 PM PT) still needs only: staffed-window push (local main ahead of origin + modified/untracked files) + the Devpost submission itself. No push attempted this run.
