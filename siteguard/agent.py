"""The agentic loop: perception -> risk reasoning -> decision -> action.

Decision policy (vision output changes downstream decisions):
  risk < 40            -> DISMISS (log only)
  40 <= risk < 80      -> REQUEST human approval; the request stays open while
                          frames stream in. Later vision can change it:
                            * hardhat appears  -> WITHDRAW the request
                            * violation persists -> ESCALATE
  risk >= 80           -> ESCALATE immediately (save evidence, emit alert)

Human-in-the-loop: approve() / reject() on a pending incident executes or
drops the action. Every step is appended to a JSON trace.
"""
import itertools
import time
from dataclasses import dataclass, field

from .config import PolicyConfig
from .perception import Track

DECISION_DISMISS = "dismiss"
DECISION_REQUEST_APPROVAL = "request_approval"
DECISION_ESCALATE = "escalate"

STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_ESCALATED = "escalated"
STATUS_DISMISSED = "dismissed"
STATUS_WITHDRAWN = "withdrawn"


@dataclass
class Incident:
    incident_id: str
    track_id: int
    kinds: list
    severity: float
    status: str = STATUS_PENDING
    frames: list = field(default_factory=list)  # frame indices observed
    risk_history: list = field(default_factory=list)
    decision_history: list = field(default_factory=list)
    created_frame: int = 0
    resolved_frame: int | None = None
    reviewer: str | None = None
    evidence_path: str | None = None


class SafetyAgent:
    def __init__(self, policy: PolicyConfig, on_action=None):
        self.policy = policy
        self.incidents: dict[str, Incident] = {}
        self.open_by_track: dict[int, str] = {}  # track_id -> incident_id
        self.trace: list[dict] = []
        self._ids = itertools.count(1)
        self.on_action = on_action or (lambda kind, payload: None)
        self.repeat_offenders: dict[int, int] = {}  # track-ish key -> frame

    # ------------------------------------------------------------------ trace
    def log(self, event: str, frame: int, **details):
        self.trace.append(
            {"t": time.time(), "frame": frame, "event": event, **details}
        )

    # ------------------------------------------------------------------ risk
    def risk_score(self, track: Track, frame: int) -> tuple[float, list[str]]:
        p = self.policy
        reasons, score = [], 0.0
        if track.hardhat_ratio < 0.5 and track.frames_seen >= p.min_track_frames:
            score += p.w_no_hardhat
            reasons.append("no_hardhat")
        if track.zone_frames > 0:
            score += p.w_zone_intrusion
            reasons.append("restricted_zone")
        if track.violation_frames >= p.persistence_escalation_frames:
            score += p.w_persistence
            reasons.append("persistent")
        if track.motion and track.motion > 1.5:
            score += p.w_motion
            reasons.append("motion")
        if track.track_id in self.repeat_offenders:
            score += p.w_repeat_offender
            reasons.append("repeat_offender")
        return min(100.0, score), reasons

    def _decide(self, risk: float) -> str:
        if risk >= self.policy.escalate_at_or_above:
            return DECISION_ESCALATE
        if risk >= self.policy.dismiss_below:
            return DECISION_REQUEST_APPROVAL
        return DECISION_DISMISS

    # ------------------------------------------------------------------ loop
    def observe(self, tracks: list[Track], frame: int, frame_image=None):
        """One agent step over the current tracks. Returns new/updated incidents."""
        changed = []
        active_tracks = {t.track_id for t in tracks}
        for track in tracks:
            risk, reasons = self.risk_score(track, frame)
            decision = self._decide(risk)
            self.log(
                "perception",
                frame,
                track_id=track.track_id,
                bbox=[int(v) for v in track.bbox],
                hardhat_ratio=round(track.hardhat_ratio, 2),
                zone_frames=track.zone_frames,
                violation_frames=track.violation_frames,
                risk=round(risk, 1),
                reasons=reasons,
                decision=decision,
            )
            inc_id = self.open_by_track.get(track.track_id)
            if decision == DECISION_DISMISS:
                if inc_id:  # vision cleared -> withdraw the open request
                    inc = self.incidents[inc_id]
                    if inc.status == STATUS_PENDING:
                        inc.status = STATUS_WITHDRAWN
                        inc.resolved_frame = frame
                        inc.decision_history.append(
                            {"frame": frame, "decision": "withdraw",
                             "note": "vision cleared: PPE compliant"}
                        )
                        self.log("decision_change", frame,
                                 incident_id=inc_id,
                                 change="request_approval -> withdrawn",
                                 cause="vision: hardhat detected / zone exited")
                        del self.open_by_track[track.track_id]
                        changed.append(inc)
                continue
            if inc_id is None:
                inc = Incident(
                    incident_id=f"INC-{next(self._ids):04d}",
                    track_id=track.track_id,
                    kinds=list(reasons),
                    severity=risk,
                    created_frame=frame,
                )
                self.incidents[inc.incident_id] = inc
                self.open_by_track[track.track_id] = inc.incident_id
                inc_id = inc.incident_id
                self.log("incident_opened", frame, incident_id=inc_id,
                         kinds=reasons, severity=round(risk, 1))
            inc = self.incidents[inc_id]
            inc.frames.append(frame)
            inc.risk_history.append(round(risk, 1))
            inc.severity = max(inc.severity, risk)
            inc.kinds = sorted(set(inc.kinds) | set(reasons))
            if decision == DECISION_ESCALATE and inc.status == STATUS_PENDING:
                # Vision persistence changed the decision: request -> escalate.
                # The incident stays mapped to its track (unlike a withdrawn
                # one) so later frames extend it instead of opening a new
                # incident + re-firing the alert every frame.
                inc.status = STATUS_ESCALATED
                inc.resolved_frame = frame
                inc.decision_history.append(
                    {"frame": frame, "decision": "escalate",
                     "note": f"risk {risk:.0f} >= threshold "
                             f"{self.policy.escalate_at_or_above:.0f}"}
                )
                self.log("decision_change", frame, incident_id=inc_id,
                         change="request_approval -> escalate",
                         cause="vision: persistent violation / zone intrusion")
                self._execute_action(inc, frame, frame_image, auto=True)
                changed.append(inc)
            elif decision == DECISION_REQUEST_APPROVAL and inc.status == STATUS_PENDING:
                inc.decision_history.append(
                    {"frame": frame, "decision": "request_approval",
                     "note": f"risk {risk:.0f} in review band"})
                self.log("approval_requested", frame, incident_id=inc_id,
                         kinds=inc.kinds, severity=round(risk, 1))
                changed.append(inc)
        # Tracks that vanished: close their open incidents as dismissed.
        for tid, inc_id in list(self.open_by_track.items()):
            if tid not in active_tracks:
                inc = self.incidents[inc_id]
                if inc.status == STATUS_PENDING:
                    inc.status = STATUS_DISMISSED
                    inc.resolved_frame = frame
                    self.log("decision_change", frame, incident_id=inc_id,
                             change="request_approval -> dismissed",
                             cause="track lost")
                del self.open_by_track[tid]
        return changed

    # ------------------------------------------------------------------ human
    def approve(self, incident_id: str, reviewer: str = "safety-officer", frame_image=None):
        inc = self.incidents.get(incident_id)
        if not inc or inc.status != STATUS_PENDING:
            return None
        inc.status = STATUS_APPROVED
        inc.reviewer = reviewer
        self.log("human_decision", -1, incident_id=incident_id,
                 decision="approved", reviewer=reviewer)
        self._execute_action(inc, -1, frame_image, auto=False)
        self.open_by_track.pop(inc.track_id, None)
        return inc

    def reject(self, incident_id: str, reviewer: str = "safety-officer", reason: str = ""):
        inc = self.incidents.get(incident_id)
        if not inc or inc.status != STATUS_PENDING:
            return None
        inc.status = STATUS_REJECTED
        inc.reviewer = reviewer
        self.repeat_offenders[inc.track_id] = 1  # remember: leniency may repeat
        self.log("human_decision", -1, incident_id=incident_id,
                 decision="rejected", reviewer=reviewer, reason=reason)
        self.open_by_track.pop(inc.track_id, None)
        return inc

    # ------------------------------------------------------------------ action
    def _execute_action(self, inc: Incident, frame: int, frame_image, auto: bool):
        payload = {
            "incident_id": inc.incident_id,
            "track_id": inc.track_id,
            "kinds": inc.kinds,
            "severity": round(inc.severity, 1),
            "auto": auto,
            "frames": inc.frames[-5:],
        }
        if frame_image is not None:
            payload["evidence_path"] = self._save_evidence(inc, frame_image)
            inc.evidence_path = payload["evidence_path"]
        self.log("action_executed", frame, incident_id=inc.incident_id,
                 action="safety_alert", auto=auto, payload=payload)
        self.on_action("safety_alert", payload)

    def _save_evidence(self, inc: Incident, frame_image) -> str:
        import os
        import cv2
        os.makedirs("assets/evidence", exist_ok=True)
        path = f"assets/evidence/{inc.incident_id}.jpg"
        cv2.imwrite(path, frame_image)
        return path

    # ------------------------------------------------------------------ query
    def pending(self) -> list[Incident]:
        return [i for i in self.incidents.values() if i.status == STATUS_PENDING]

    def summary(self) -> dict:
        counts = {}
        for i in self.incidents.values():
            counts[i.status] = counts.get(i.status, 0) + 1
        return {"incidents": len(self.incidents), "by_status": counts,
                "trace_events": len(self.trace)}
