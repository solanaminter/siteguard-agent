# STORYBOARD — SiteGuard Agent demo video
---
flow: automation
storyboard: no
message: "SiteGuard Agent — an agentic vision system where OpenCV 5 perception drives safety decisions that later vision can change"
aspect: 1920x1080
length: 229s
---

## Frame 1 — scene1_problem (0–32s)
status: rendered
src: compositions/scene1_problem.html
blueprint: kinetic-type-beats
beat: "Construction injuries are preventable. Fixed alarms cry wolf. SiteGuard watches, reasons, and changes its mind."
visual: dark blueprint grid; title "SiteGuard Agent" assembles; problem stats type in beats; construction still backdrop (compliant.webp, dimmed).

## Frame 2 — scene2_demo (32–86s)
status: rendered
src: compositions/scene2_demo.html
blueprint: (footage-led; minimal motion, caption callouts)
beat: "The real pipeline: detection → hardhat segmentation → zones → tracking → risk → decisions that change."
visual: demo_annotated.mp4 at 0.55x (49s) full-bleed; lower-third captions name each beat (DISMISS / REQUEST APPROVAL / WITHDRAWN / ESCALATE / APPROVED); risk bar animates.

## Frame 3 — scene3_trace (86–116s)
status: rendered
src: compositions/scene3_trace.html
blueprint: grid-card-assemble
beat: "The trace is the proof: incident opened → withdrawn by later vision → escalated by persistence."
visual: trace event cards cascade in timeline order; decision_change events highlighted amber/red; final line "vision changed what the system did."

## Frame 4 — scene4_arch (116–147s)
status: rendered
src: compositions/scene4_arch.html
blueprint: (slow Ken Burns over diagram)
beat: "OpenCV 5 perception → agent loop → AWS: Fargate + ALB, S3 evidence, CloudWatch, CloudFormation."
visual: ARCHITECTURE.png slow zoom; section labels pulse in sequence (PERCEPTION → AGENT LOOP → AWS).

## Frame 5 — scene5_eval (147–188s)
status: rendered
src: compositions/scene5_eval.html
blueprint: grid-card-assemble
beat: "Measured honestly: 0.905 detection, 1.00 hardhat precision, 1.00 zone accuracy — and documented failures."
visual: metric cards count up; failure-case stills (whitehat_sky, occlusion) with honest captions.

## Frame 6 — scene6_responsible (188–214s)
status: rendered
src: compositions/scene6_responsible.html
blueprint: kinetic-type-beats
beat: "Decision support, not enforcer. Human in the loop. No facial recognition. Never wired to punishment."
visual: centered statements build beat by beat on blueprint grid.

## Frame 7 — scene7_credits (214–229s)
status: rendered
src: compositions/scene7_credits.html
blueprint: logo-assemble-lockup (text lockup variant)
beat: "Built by Dy (Solana Minter) / Green Robin for the OpenCV AI Competition 2026."
visual: "SiteGuard Agent" lockup assembles; credit lines fade in; hold.
