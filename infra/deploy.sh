#!/usr/bin/env bash
# One-command deploy of SiteGuard Agent to AWS (ECS Fargate + S3 + CloudWatch).
# Prerequisites: AWS CLI configured with a profile that can create ECR/ECS/
# CloudFormation resources, and docker with buildx.
#
# Usage:
#   AWS_PROFILE=myprofile ./infra/deploy.sh <vpc-id> <subnet-id-1,subnet-id-2>
set -euo pipefail

REGION="${AWS_REGION:-us-west-2}"
REPO="siteguard"
STACK="siteguard-agent"
VPC_ID="${1:?usage: deploy.sh <vpc-id> <subnet-ids-csv>}"
SUBNETS="${2:?usage: deploy.sh <vpc-id> <subnet-ids-csv>}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
ECR="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO"

echo "==> ECR login"
aws ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO" >/dev/null
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ECR" >/dev/null

echo "==> Build & push image"
docker build -f infra/Dockerfile -t "$ECR:latest" .
docker push "$ECR:latest"

echo "==> Deploy CloudFormation stack"
aws cloudformation deploy \
  --stack-name "$STACK" \
  --template-file infra/cloudformation.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides ImageUri="$ECR:latest" VpcId="$VPC_ID" SubnetIds="$SUBNETS" \
  --region "$REGION"

echo "==> Done. Tail logs with:"
echo "aws logs tail /ecs/siteguard-agent --follow --region $REGION"
echo "==> Public endpoint:"
aws cloudformation describe-stacks --stack-name "$STACK" --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='EndpointURL'].OutputValue" --output text
