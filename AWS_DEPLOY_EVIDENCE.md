# SiteGuard Agent — AWS deployment evidence

**Date:** 2026-09-14 (PDT)
**Region:** us-west-2
**Account:** 899562467628

## What was deployed

Real, meaningful AWS deployment of the SiteGuard Agent container via
`infra/cloudformation.yaml`:

- **ECR** `siteguard` — image built with AWS CodeBuild (project `siteguard-build`),
  pushed as `899562467628.dkr.ecr.us-west-2.amazonaws.com/siteguard:latest`.
  Image contains the OpenCV 5 headless detection pipeline (YOLOX-s ONNX via
  `cv2.dnn`) + FastAPI service (`siteguard.app`).
- **ECS Fargate** service `siteguard-agent` (1 task, `awsvpc`, public IP).
- **Application Load Balancer** with listener → target group → ECS tasks.
- **S3** `siteguard-evidence-899562467628` — versioned evidence bucket.
- **CloudWatch Logs** `/ecs/siteguard-agent` (30-day retention).
- **IAM** task + execution roles created by the stack.

## Live verification (before teardown)

- Stack status: `CREATE_COMPLETE`
- Public endpoint: `http://siteguard-alb-1102344515.us-west-2.elb.amazonaws.com`
- `GET /health` → **HTTP 200** `{"status":"ok","service":"siteguard-agent"}`

## Teardown

Stack deleted after verification to stop ALB/Fargate billing. The ECR image,
CodeBuild project, and build role were kept so the deployment can be recreated
with a single `create_stack` call (template fix: `Service` now has
`DependsOn: Listener`, fixing the original "target group has no associated
load balancer" failure).
