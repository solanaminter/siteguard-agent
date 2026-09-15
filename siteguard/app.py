"""FastAPI demo service: analyze images/video, review incidents, human approval.

Run locally:
    uvicorn siteguard.app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import tempfile
import uuid

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .agent import SafetyAgent
from .annotate import annotate
from .config import AppConfig
from .observability import JsonLogger
from .perception import PerceptionPipeline
from .storage import get_storage

cfg = AppConfig(
    storage_backend=os.environ.get("STORAGE_BACKEND", "local"),
    s3_bucket=os.environ.get("S3_BUCKET", "siteguard-evidence"),
    aws_region=os.environ.get("AWS_REGION", "us-west-2"),
)
pipeline = PerceptionPipeline(cfg, cfg.zones)
storage = get_storage(cfg.storage_backend, cfg.s3_bucket, cfg.aws_region)
log = JsonLogger()
app = FastAPI(title="SiteGuard Agent", version="1.0.0")

_agents: dict[str, SafetyAgent] = {}


def get_agent(session: str) -> SafetyAgent:
    if session not in _agents:
        actions = []

        def on_action(kind, payload):
            actions.append({"kind": kind, "payload": payload})
            log.info("action_dispatched", kind=kind, **payload)

        _agents[session] = SafetyAgent(cfg.policy, on_action=on_action)
        _agents[session].dispatched_actions = actions
    return _agents[session]


def _incident_dict(inc):
    return {
        "incident_id": inc.incident_id,
        "track_id": inc.track_id,
        "kinds": inc.kinds,
        "severity": round(inc.severity, 1),
        "status": inc.status,
        "frames": inc.frames,
        "risk_history": inc.risk_history,
        "decision_history": inc.decision_history,
        "reviewer": inc.reviewer,
        "evidence": inc.evidence_path,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"status": "ok", "service": "siteguard-agent"}


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...),
                        session: str = Form("default")):
    data = await file.read()
    arr = cv2.imdecode(__import__("numpy").frombuffer(data, __import__("numpy").uint8),
                       cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, "unreadable image")
    agent = get_agent(session)
    detections, tracks, motion = pipeline.process(arr, 0)
    agent.observe(tracks, 0, frame_image=arr)
    annotated = annotate(arr.copy(), detections, tracks, cfg.zones)
    out = f"/tmp/siteguard_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(out, annotated)
    log.info("image_analyzed", session=session,
             detections=len(detections), incidents=len(agent.incidents))
    return {
        "detections": [
            {"bbox": [float(v) for v in d.bbox],
             "confidence": round(float(d.confidence), 3),
             "hardhat": bool(d.hardhat),
             "hardhat_confidence": round(float(d.hardhat_confidence), 3),
             "in_restricted_zone": bool(d.in_restricted_zone)}
            for d in detections
        ],
        "incidents": [_incident_dict(i) for i in agent.incidents.values()],
        "annotated_image": out,
    }


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...),
                        session: str = Form("default"),
                        max_frames: int = Form(240),
                        stride: int = Form(2)):
    tmp = tempfile.mktemp(suffix=".mp4")
    with open(tmp, "wb") as f:
        f.write(await file.read())
    cap = cv2.VideoCapture(tmp)
    agent = get_agent(session)
    n = 0
    frames = []
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % stride:
            continue
        frame = cv2.resize(frame, (cfg.frame_width,
                                   int(frame.shape[0] * cfg.frame_width / frame.shape[1])))
        detections, tracks, motion = pipeline.process(frame, n)
        agent.observe(tracks, n, frame_image=frame)
        frames.append((frame, detections, tracks))
    cap.release()
    # Render annotated video.
    out = f"/tmp/siteguard_{uuid.uuid4().hex}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out, fourcc, 15,
                         (frames[0][0].shape[1], frames[0][0].shape[0])) if frames else None
    for frame, detections, tracks in frames:
        vw.write(annotate(frame.copy(), detections, tracks, cfg.zones))
    if frames:
        vw.release()
    log.info("video_analyzed", session=session, frames=n,
             incidents=len(agent.incidents))
    return {
        "frames_processed": n,
        "incidents": [_incident_dict(i) for i in agent.incidents.values()],
        "summary": agent.summary(),
        "annotated_video": out if frames else None,
    }


@app.get("/api/incidents")
def list_incidents(session: str = "default"):
    agent = get_agent(session)
    return {"incidents": [_incident_dict(i) for i in agent.incidents.values()],
            "pending": [i.incident_id for i in agent.pending()]}


@app.post("/api/incidents/{incident_id}/approve")
def approve_incident(incident_id: str, session: str = "default",
                     reviewer: str = "safety-officer"):
    agent = get_agent(session)
    inc = agent.approve(incident_id, reviewer=reviewer)
    if not inc:
        raise HTTPException(404, "incident not found or not pending")
    log.info("incident_approved", incident_id=incident_id, reviewer=reviewer)
    return {"incident": _incident_dict(inc)}


@app.post("/api/incidents/{incident_id}/reject")
def reject_incident(incident_id: str, session: str = "default",
                    reviewer: str = "safety-officer", reason: str = ""):
    agent = get_agent(session)
    inc = agent.reject(incident_id, reviewer=reviewer, reason=reason)
    if not inc:
        raise HTTPException(404, "incident not found or not pending")
    log.info("incident_rejected", incident_id=incident_id, reviewer=reviewer)
    return {"incident": _incident_dict(inc)}


@app.get("/api/trace")
def get_trace(session: str = "default", event: str | None = None):
    agent = get_agent(session)
    trace = agent.trace
    if event:
        trace = [e for e in trace if e["event"] == event]
    return {"trace": trace, "count": len(trace)}


DASHBOARD_HTML = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>SiteGuard Agent — Safety Review Console</title>
<style>
body{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}
h1{font-size:22px} .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px;margin:12px 0}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;margin-right:6px}
.pending{background:#9e6a03}.approved{background:#1a7f37}.rejected{background:#5a5a5a}
.escalated{background:#da3633}.dismissed{background:#333}.withdrawn{background:#6e40c9}
button{background:#1f6feb;color:#fff;border:0;border-radius:6px;padding:8px 14px;margin-right:8px;cursor:pointer}
button.reject{background:#6e7681}
pre{background:#0d1117;padding:12px;border-radius:6px;overflow:auto;font-size:12px}
</style></head><body>
<h1>&#x1F6A7; SiteGuard Agent — Safety Review Console</h1>
<p>Agentic vision loop: OpenCV 5 perception &rarr; risk reasoning &rarr; decision &rarr; human approval &rarr; action.</p>
<div class="card"><h3>Analyze</h3>
<form id="upform"><input type="file" id="file" accept="image/*,video/*">
<button type="submit">Analyze</button></form><div id="result"></div></div>
<div class="card"><h3>Incidents <button onclick="load()">Refresh</button></h3><div id="incidents"></div></div>
<div class="card"><h3>Agent trace (decision changes)</h3><pre id="trace"></pre></div>
<script>
async function load(){
  const r=await fetch('/api/incidents'); const j=await r.json();
  document.getElementById('incidents').innerHTML=j.incidents.map(i=>
   `<div class="card"><span class="badge ${i.status}">${i.status}</span>
    <b>${i.incident_id}</b> track #${i.track_id} &mdash; severity ${i.severity}<br>
    kinds: ${i.kinds.join(', ')}<br>
    <button onclick="review('${i.incident_id}','approve')">Approve &amp; alert</button>
    <button class="reject" onclick="review('${i.incident_id}','reject')">Reject</button>
    ${i.evidence?`<br>evidence: ${i.evidence}`:''}</div>`).join('')||'<i>none</i>';
  const t=await fetch('/api/trace?event=decision_change'); const tj=await t.json();
  document.getElementById('trace').textContent=JSON.stringify(tj.trace,null,1);
}
async function review(id,how){
  await fetch(`/api/incidents/${id}/${how}`,{method:'POST'}); load();
}
document.getElementById('upform').onsubmit=async e=>{
  e.preventDefault();
  const f=document.getElementById('file').files[0]; if(!f) return;
  const fd=new FormData(); fd.append('file',f);
  const isImg=f.type.startsWith('image');
  const r=await fetch(isImg?'/api/analyze/image':'/api/analyze/video',{method:'POST',body:fd});
  const j=await r.json();
  document.getElementById('result').innerHTML='<pre>'+JSON.stringify(j.incidents||j.summary,null,1)+'</pre>';
  load();
};
load();
</script></body></html>"""
