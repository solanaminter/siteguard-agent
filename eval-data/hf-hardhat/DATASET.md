# HF hard-hat evaluation subset

Real-world labeled evaluation data for the SiteGuard perception layer,
replacing hand-labeled synthetic stills with genuine construction-site
imagery.

## Source

- Hugging Face: `keremberke/hard-hat-detection` (split: `data/valid.zip`)
- Upstream: Roboflow Universe "Hard Hats" project
  (`roboflow-universe-projects/hard-hats-fhbh5`), exported 2023-01-16
- License: **CC BY 4.0** (per the dataset's `README.dataset.txt`)
- Full valid split: 3,962 images, 11,174 head annotations
  (`hardhat` = 8,952 / `no-hardhat` = 2,222), COCO format, 640x640

## This subset

- 200 images, deterministically sampled (seed `20260915`):
  - 100 images containing at least one `no-hardhat` head
  - 100 hardhat-only images
- Resulting annotation balance: 297 hardhat / 207 no-hardhat heads
- `annotations.json`: COCO annotations filtered to the sampled images
- `images/`: the 200 sampled images (~11 MB)

## Reproduce

```bash
# download (225 MB)
curl -L -o hf-hardhat-valid.zip \
  https://huggingface.co/datasets/keremberke/hard-hat-detection/resolve/main/data/valid.zip
# the sampling procedure is documented in the build log; re-running
# siteguard/eval_hf.py --data-dir eval-data/hf-hardhat reproduces the metrics
```

## Attribution

Hard-hat annotations by the Roboflow Universe community contributors to the
"Hard Hats" dataset, used under CC BY 4.0. Person boxes are produced by
SiteGuard's own detector at eval time (the source labels heads, not persons).
