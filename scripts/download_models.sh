#!/usr/bin/env bash
# Fetch the pinned detection model (OpenCV Zoo YOLOX-s, Apache 2.0) and
# verify its SHA256. The model is vendored in the repo at
# assets/models/yolox_s_2022nov.onnx; this script re-downloads it if missing.
set -euo pipefail
cd "$(dirname "$0")/../assets/models"

MODEL="yolox_s_2022nov.onnx"
URL="https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx"

if [ ! -f "$MODEL" ]; then
  echo "downloading $MODEL ..."
  curl -sSL --retry 3 -o "$MODEL" "$URL"
fi
sha256sum -c SHA256SUMS
echo "model OK: $MODEL"
