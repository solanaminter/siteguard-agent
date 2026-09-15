"""Central configuration for the SiteGuard agent."""
from dataclasses import dataclass, field


@dataclass
class PerceptionConfig:
    # cv2.dnn person detector (OpenCV Zoo YOLOX-s ONNX) settings.
    # Empty string -> PersonDetector resolves the vendored model path.
    model_path: str = ""
    dnn_conf_thresh: float = 0.35
    dnn_nms_thresh: float = 0.5
    dnn_input_size: int = 640
    # Hardhat HSV segmentation: fraction of the head ROI that must be
    # hardhat-colored pixels to count as a hardhat.
    hardhat_min_coverage: float = 0.22
    # Optical-flow motion magnitude threshold for "activity" flag.
    flow_motion_threshold: float = 1.5


@dataclass
class PolicyConfig:
    # Frames a track must persist before it is trusted.
    min_track_frames: int = 3
    # Frames of continuous violation before a medium-risk incident escalates.
    persistence_escalation_frames: int = 45
    # Risk thresholds.
    dismiss_below: float = 40.0
    escalate_at_or_above: float = 80.0
    # Risk point contributions.
    w_no_hardhat: float = 40.0
    w_zone_intrusion: float = 35.0
    w_persistence: float = 10.0
    w_motion: float = 5.0
    w_repeat_offender: float = 15.0
    # A track dismissed before counts as a repeat offender on re-violation.
    repeat_offender_memory: int = 600  # frames


@dataclass
class Zone:
    name: str
    # Polygon in relative (0..1) coordinates.
    polygon: list
    restricted: bool = True


@dataclass
class AppConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    # Default camera zone: lower-right quadrant marked restricted.
    zones: list = field(
        default_factory=lambda: [
            Zone(
                name="excavation-zone",
                polygon=[(0.55, 0.45), (0.98, 0.45), (0.98, 0.98), (0.55, 0.98)],
                restricted=True,
            )
        ]
    )
    frame_width: int = 1280
    evidence_dir: str = "assets/evidence"
    storage_backend: str = "local"  # or "s3"
    s3_bucket: str = "siteguard-evidence"
    aws_region: str = "us-west-2"
