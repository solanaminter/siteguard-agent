"""Agent policy tests — no CV required, synthetic tracks."""
from siteguard.agent import (
    SafetyAgent, STATUS_APPROVED, STATUS_ESCALATED, STATUS_PENDING,
    STATUS_REJECTED, STATUS_WITHDRAWN,
)
from siteguard.config import PolicyConfig
from siteguard.perception import Track


def make_track(tid=1, hardhat_votes=(False,) * 5, zone_frames=0,
               violation_frames=5, frames_seen=5, motion=0.0):
    t = Track(track_id=tid, bbox=(10, 10, 60, 140), frames_seen=frames_seen)
    t.hardhat_votes = list(hardhat_votes)
    t.zone_frames = zone_frames
    t.violation_frames = violation_frames
    t.motion = motion
    return t


def test_low_risk_dismissed():
    agent = SafetyAgent(PolicyConfig())
    t = make_track(hardhat_votes=(True,) * 5, violation_frames=0)
    agent.observe([t], frame=1)
    assert agent.summary()["incidents"] == 0


def test_medium_risk_requests_approval():
    agent = SafetyAgent(PolicyConfig())
    agent.observe([make_track()], frame=1)
    pending = agent.pending()
    assert len(pending) == 1 and pending[0].status == STATUS_PENDING
    assert "no_hardhat" in pending[0].kinds


def test_high_risk_auto_escalates():
    agent = SafetyAgent(PolicyConfig())
    t = make_track(zone_frames=3, violation_frames=50)
    agent.observe([t], frame=1)
    incs = list(agent.incidents.values())
    assert incs[0].status == STATUS_ESCALATED


def test_vision_clears_withdraws_request():
    """The Agentic Vision loop: later vision output changes the decision."""
    agent = SafetyAgent(PolicyConfig())
    bad = make_track()
    agent.observe([bad], frame=1)
    assert len(agent.pending()) == 1
    good = make_track(hardhat_votes=(True,) * 5, violation_frames=0)
    agent.observe([good], frame=2)
    incs = list(agent.incidents.values())
    assert incs[0].status == STATUS_WITHDRAWN
    assert any(e["event"] == "decision_change" for e in agent.trace)


def test_persistence_escalates():
    agent = SafetyAgent(PolicyConfig(persistence_escalation_frames=45))
    # medium risk: 40 + 10 persistence = 50 -> stays pending for a while...
    # force escalation by adding zone intrusion after some frames
    t = make_track(violation_frames=10)
    agent.observe([t], frame=1)
    assert agent.pending()[0].status == STATUS_PENDING
    t2 = make_track(zone_frames=5, violation_frames=50)
    agent.observe([t2], frame=60)
    assert list(agent.incidents.values())[0].status == STATUS_ESCALATED


def test_human_approve_executes_action():
    fired = []
    agent = SafetyAgent(PolicyConfig(), on_action=lambda k, p: fired.append((k, p)))
    agent.observe([make_track()], frame=1)
    inc = agent.pending()[0]
    agent.approve(inc.incident_id, reviewer="dy")
    assert inc.status == STATUS_APPROVED and inc.reviewer == "dy"
    assert fired and fired[0][0] == "safety_alert"
    assert any(e["event"] == "human_decision" for e in agent.trace)


def test_human_reject():
    agent = SafetyAgent(PolicyConfig())
    agent.observe([make_track()], frame=1)
    inc = agent.pending()[0]
    agent.reject(inc.incident_id, reviewer="dy", reason="false positive")
    assert inc.status == STATUS_REJECTED
    assert len(agent.pending()) == 0


def test_trace_records_perception_to_action():
    agent = SafetyAgent(PolicyConfig())
    agent.observe([make_track(zone_frames=3, violation_frames=50)], frame=7)
    events = [e["event"] for e in agent.trace]
    assert "perception" in events and "action_executed" in events


def test_escalated_track_does_not_reopen_incident():
    """An escalated track must not spawn a new incident + alert every frame."""
    fired = []
    agent = SafetyAgent(PolicyConfig(),
                        on_action=lambda k, p: fired.append((k, p)))
    t = make_track(zone_frames=3, violation_frames=50)
    agent.observe([t], frame=1)
    assert list(agent.incidents.values())[0].status == STATUS_ESCALATED
    assert len(fired) == 1
    # Same track keeps violating: incident extends, no new incident, no new alert.
    agent.observe([t], frame=2)
    agent.observe([t], frame=3)
    assert len(agent.incidents) == 1
    assert len(fired) == 1
    assert list(agent.incidents.values())[0].frames == [1, 2, 3]
