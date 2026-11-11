"""Bharat-TTC Layer 1: lane-free calibration, class-specific CTRV tracking and 2D conflict scoring."""
__version__ = "0.3.0"  # keep in sync with pyproject.toml (checked by tests)
from .camera import (
    CALIBRATION_WIDTH,
    VEHICLE_SIZE,
    Calibration,
    CameraModel,
    bbox_bottom_centre,
    calibrate_from_detections,
    calibrate_from_vehicles,
    camera_to_world,
    rotate_to_camera,
    simulate_calibration_traffic,
    world_to_camera,
)
from .motion_models import (
    CLASS_AGILITY,
    ActorClass,
    ClassAgility,
    EgoState,
    ctrv_step,
    estimate_agility,
    noise_gain,
    process_noise,
    wrap_angle,
)
from .risk import (
    DEFAULT_EGO_RADIUS,
    AlertEvent,
    AlertPolicy,
    RiskAssessment,
    assess,
    combined_radius,
    conflict_probability,
    crawl_gate,
    predicted_paths,
    rollout,
    side_of,
    track_ttc_2d,
    ttc_1d,
    ttc_2d,
)
from .ukf import CTRVUKF
