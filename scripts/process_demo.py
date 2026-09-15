"""Run the full agentic demo scenario over assets/annotated/demo_input.mp4.

Scenario (frame ranges refer to the stitched input):
  seg1 compliant            -> dismiss (log only)
  seg2 violation_nohat      -> INC request_approval (pending)
  seg3 violation_corrected  -> vision clears  -> WITHDRAWN   [decision change]
  seg4 restricted_zone      -> INC -> ESCALATE (auto)        [decision change]
  seg5 violation_nohat      -> INC request_approval -> officer APPROVES -> action

Writes:
  assets/annotated/demo_annotated.mp4   annotated footage for the demo video
  assets/annotated/trace.json           full agent trace (drives video overlays)
  docs/demo_summary.json                incident summary
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from siteguard.agent import SafetyAgent
from siteguard.annotate import annotate
from siteguard.config import AppConfig, Zone
from siteguard.observability import JsonLogger
from siteguard.perception import PerceptionPipeline

IN = "assets/annotated/demo_input.mp4"
OUT = "assets/annotated/demo_annotated.mp4"
FPS = 15

# Per-segment zone configs: (start_frame, end_frame, zones)
SEG_ZONES = [
    (0, 75, [Zone("yard-corner", [[0.78, 0.0], [1.0, 0.0], [1.0, 0.3], [0.78, 0.3]])]),
    (75, 150, []),
    (150, 225, []),
    (225, 330, [Zone("excavation", [[0.2, 0.35], [1.0, 0.2], [1.0, 0.95], [0.2, 0.95]])]),
    (330, 405, []),
]

log = JsonLogger()


def zones_for(frame_idx):
    for s, e, z in SEG_ZONES:
        if s <= frame_idx < e:
            return z
    return []


def main():
    cfg = AppConfig()
    actions = []
    agent = SafetyAgent(cfg.policy,
                        on_action=lambda k, p: actions.append({"kind": k, "payload": p}))
    cap = cv2.VideoCapture(IN)
    assert cap.isOpened(), IN
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (1280, 720))
    pipe = None
    n = 0
    approved_inc = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if pipe is None:
            pipe = PerceptionPipeline(cfg, zones_for(n))
        else:
            pipe.zones.zones = zones_for(n)
        detections, tracks, motion = pipe.process(frame, n)
        agent.observe(tracks, n, frame_image=frame)
        # Status banner from the latest agent state.
        pending = agent.pending()
        if any(i.status == "escalated" for i in agent.incidents.values()):
            banner = "!! ESCALATED: safety alert dispatched"
        elif pending:
            banner = f"AWAITING HUMAN REVIEW: {pending[0].incident_id} " \
                     f"({', '.join(pending[0].kinds)})"
        else:
            banner = None
        vw.write(annotate(frame, detections, tracks, zones_for(n), banner))
        if n % 30 == 0:
            log.info("demo_progress", frame=n,
                     tracks=[t.track_id for t in tracks],
                     pending=[i.incident_id for i in pending])
        n += 1
    cap.release()
    vw.release()

    # Human-in-the-loop: officer approves the last pending incident.
    pending = agent.pending()
    if pending:
        approved_inc = agent.approve(pending[0].incident_id,
                                    reviewer="safety-officer")
        log.info("demo_human_approval", incident_id=approved_inc.incident_id)

    os.makedirs("docs", exist_ok=True)
    with open("assets/annotated/trace.json", "w") as f:
        json.dump(agent.trace, f, indent=1)
    summary = {
        "frames": n,
        "incidents": [
            {"id": i.incident_id, "track": i.track_id, "kinds": i.kinds,
             "severity": round(i.severity, 1), "status": i.status,
             "created_frame": i.created_frame, "resolved_frame": i.resolved_frame,
             "reviewer": i.reviewer, "evidence": i.evidence_path,
             "decision_history": i.decision_history}
            for i in agent.incidents.values()
        ],
        "actions": actions,
        "trace_events": len(agent.trace),
    }
    with open("docs/demo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
