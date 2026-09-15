"""Render ARCHITECTURE.png with OpenCV drawing primitives (blueprint theme)."""
import cv2
import numpy as np

W, H = 1920, 1080
BG = (23, 17, 13)
BOX = (34, 27, 22)
BORDER = (61, 54, 48)
WHITE = (237, 237, 230)
DIM = (160, 160, 160)
OCV = (70, 130, 255)
AWS = (0, 153, 255)
AGENT = (113, 204, 46)

img = np.zeros((H, W, 3), dtype=np.uint8)
img[:] = BG


def box(x, y, w, h, title, lines, accent=WHITE):
    cv2.rectangle(img, (x, y), (x + w, y + h), BOX, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 2)
    cv2.rectangle(img, (x, y), (x + w, y + 40), BORDER, -1)
    cv2.putText(img, title, (x + 14, y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, accent, 2)
    yy = y + 68
    for line in lines:
        cv2.putText(img, line, (x + 14, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, WHITE, 1)
        yy += 28


def arrow(x1, y1, x2, y2, color=DIM):
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, 2, tipLength=0.04)


def header(x, y, text, color):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)


cv2.putText(img, "SiteGuard Agent -- System Architecture", (48, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.25, WHITE, 2)
cv2.putText(img, "OpenCV 5 perception  ->  agentic decision loop  ->  AWS cloud delivery",
            (48, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.65, DIM, 2)

TOP = 140
# ---------- Column 1: ingest ----------
header(48, TOP - 14, "1  SITE / INGEST", DIM)
box(48, TOP, 500, 190, "Site cameras",
    ["RTSP / MP4 site cameras", "Frame grabber (FFmpeg)", "1280x720 @ 15 fps"], DIM)
box(48, TOP + 215, 500, 190, "Pre-processing",
    ["resize / letterbox", "BGR -> HSV / gray", "frame queue"], DIM)
arrow(298, TOP + 190, 298, TOP + 215, DIM)

# ---------- Column 2: perception ----------
header(620, TOP - 14, "2  PERCEPTION -- OpenCV 5", OCV)
PX, PW = 620, 600
box(PX, TOP, PW, 150, "Person detection  [cv2.dnn]",
    ["OpenCV Zoo YOLOX-s ONNX via readNetFromONNX", "letterbox 640 + NMS",
     "COCO class 0 (person), conf >= 0.35"], OCV)
box(PX, TOP + 172, PW, 150, "Hardhat segmentation  [imgproc]",
    ["cvtColor BGR->HSV on head ROI", "inRange yellow/orange/red/blue/white",
     "dome-shape / yellow-dominance / white-texture paths"], OCV)
box(PX, TOP + 344, PW, 150, "Zones + motion  [imgproc/video]",
    ["pointPolygonTest restricted polygons",
     "calcOpticalFlowFarneback motion signal"], OCV)
box(PX, TOP + 516, PW, 130, "Tracking",
    ["IoU centroid tracker", "15-frame hardhat majority vote"], OCV)
for a, b in [(TOP + 150, TOP + 172), (TOP + 322, TOP + 344), (TOP + 494, TOP + 516)]:
    arrow(PX + PW // 2, a, PX + PW // 2, b, OCV)

# ---------- Column 3: AWS ----------
header(1292, TOP - 14, "3  AWS CLOUD DELIVERY", AWS)
AX, AW = 1292, 580
box(AX, TOP, AW, 150, "Compute  [ECS Fargate]",
    ["Docker image from ECR", "uvicorn siteguard.app (FastAPI)",
     "1 task, 1 vCPU / 2 GB"], AWS)
box(AX, TOP + 172, AW, 150, "Storage  [S3]",
    ["siteguard-evidence-<acct> bucket", "evidence clips + annotated video",
     "versioned, 90-day lifecycle"], AWS)
box(AX, TOP + 344, AW, 150, "Observability  [CloudWatch]",
    ["structured JSON logs (1 line = 1 event)", "/ecs/siteguard-agent log group",
     "alarms on escalation rate"], AWS)
box(AX, TOP + 516, AW, 130, "Deploy",
    ["infra/deploy.sh: ECR push +", "CloudFormation (VPC, IAM)"], AWS)
for a, b in [(TOP + 150, TOP + 172), (TOP + 322, TOP + 344), (TOP + 494, TOP + 516)]:
    arrow(AX + AW // 2, a, AX + AW // 2, b, AWS)

# cross-column arrows
arrow(548, TOP + 95, PX, TOP + 95, WHITE)
arrow(PX + PW, TOP + 95, AX, TOP + 95, WHITE)
cv2.putText(img, "frames", (556, TOP + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1)
cv2.putText(img, "alerts + evidence", (PX + PW + 8, TOP + 86),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, DIM, 1)

# ---------- Agent loop band ----------
AY = 820
header(48, AY - 14, "4  AGENT LOOP  (perception -> decision -> action)", AGENT)
box(48, AY, 430, 135, "Risk scoring",
    ["no_hardhat +40, zone +35", "persistence +10, motion +5", "repeat offender +15"], AGENT)
box(508, AY, 430, 135, "Policy",
    ["< 40 dismiss (log only)", "40-79 request human approval", ">= 80 auto-escalate"], AGENT)
box(968, AY, 430, 135, "Human-in-the-loop",
    ["review console: approve /", "reject with evidence attached"], AGENT)
box(1428, AY, 444, 135, "Actions + trace",
    ["save evidence clip, safety alert", "JSON trace of every decision"], AGENT)
for xa, xb in [(478, 508), (938, 968), (1398, 1428)]:
    arrow(xa, AY + 67, xb, AY + 67, AGENT)
arrow(PX + PW // 2, TOP + 646, PX + PW // 2, AY - 30, AGENT)  # perception -> loop

cv2.putText(img, "Built by Dy (Solana Minter) / Green Robin  |  OpenCV AI Competition 2026",
            (48, H - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DIM, 1)

cv2.imwrite("ARCHITECTURE.png", img)
print("wrote ARCHITECTURE.png", img.shape)
