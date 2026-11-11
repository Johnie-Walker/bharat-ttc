"""2D conflict scoring (Layer 1, stage D).

Closed-form time-to-collision for two discs of combined radius R, with
r = p_i - p_ego and v = v_i - v_ego:

    TTC_2D = ( -r.v - sqrt((r.v)^2 - |v|^2 (|r|^2 - R^2)) ) / |v|^2

defined only when closing (r.v < 0) and the root is real. The probabilistic
score rolls the filter's sigma points forward and sums the weights of those
whose future path comes within R of the ego's:

    P_conf = sum_j w_j 1[ min_{t <= T} |p_i^(j)(t) - p_ego(t)| < R ]
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .motion_models import ClassAgility, EgoState, ctrv_step
from .ukf import N_X, CTRVUKF

DEFAULT_EGO_RADIUS = 1.4  # m; front of a truck cab, approximated as a disc

# Stage-E starting thresholds from the brief (to be tuned on validation data).
CRAWL_EGO_SPEED = 18.0 / 3.6    # m/s
CRAWL_LATERAL_SPEED = 1.2       # m/s
P_CONF_MIN = 0.5
TTC_2D_MAX = 1.8                # s
ALERT_RANGE = 35.0              # m
# Lane-based baseline: 1D TTC under 2 s inside a fixed forward corridor.
CORRIDOR_HALF_WIDTH = 1.5       # m
TTC_1D_MAX = 2.0                # s


def ttc_2d(r: np.ndarray, v: np.ndarray, R: float) -> float:
    """Time until the actors first come within R of each other; math.inf if never."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rr, rv, vv = r @ r, r @ v, v @ v
    if rr <= R * R:
        return 0.0
    if rv >= 0.0 or vv < 1e-12:
        return math.inf
    disc = rv * rv - vv * (rr - R * R)
    if disc < 0.0:
        return math.inf
    return float((-rv - math.sqrt(disc)) / vv)


def combined_radius(track: CTRVUKF, ego_radius: float = DEFAULT_EGO_RADIUS) -> float:
    return track.agility.radius + ego_radius


def track_ttc_2d(track: CTRVUKF, ego: EgoState, ego_radius: float = DEFAULT_EGO_RADIUS) -> float:
    """TTC_2D from the filter's mean state."""
    return ttc_2d(track.position - ego.position, track.velocity - ego.velocity,
                  combined_radius(track, ego_radius))


def rollout(points: np.ndarray, agility: ClassAgility, dt: float, n_steps: int) -> np.ndarray:
    """Roll augmented points [x(5), nu_a, nu_psidd] forward with their noise held constant.

    Holding nu constant turns each point into one manoeuvre (keep braking,
    keep tightening the turn). Speed and yaw rate are clamped to the class's
    physical limits. Returns positions of shape (n_steps+1, N, 2).
    """
    x = points[:, :N_X].copy()
    nu = points[:, N_X:]
    out = np.empty((n_steps + 1, x.shape[0], 2))
    out[0] = x[:, :2]
    w_max = agility.yaw_rate_max
    for k in range(1, n_steps + 1):
        x = ctrv_step(x, dt, nu)
        np.clip(x[:, 2], 0.0, agility.v_max, out=x[:, 2])
        np.clip(x[:, 4], -w_max, w_max, out=x[:, 4])
        out[k] = x[:, :2]
    return out


def conflict_probability(
    track: CTRVUKF,
    ego: EgoState,
    horizon: float = 2.5,
    dt: float = 0.1,
    ego_radius: float = DEFAULT_EGO_RADIUS,
    lam: float = 1.0,
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
    paths: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """P_conf over the horizon.

    paths: optional (paths, weights) from predicted_paths(track, horizon, dt, lam),
    to reuse one rollout for both scoring and drawing.

    The filter's noise is white: a fresh nu every step_dt. A rolled-out path
    holds one nu for the whole horizon T instead, so it gets the variance of
    the time-averaged white noise, Q * step_dt / T. Holding that constant
    reproduces the white-noise heading spread at T to within ~13% (std ratio
    sqrt(3/4)); holding the raw per-step Q would overstate it by orders of
    magnitude and paint every nearby auto as a conflict.

    By default uses the augmented sigma points, as in the document. lam must be
    >= 0 so every weight is non-negative and the sum is a valid probability
    (the filter's own lam = 3 - n gives a negative centre weight). Each of the
    15 points carries ~1/16 of the mass however far out it sits, so the result
    is coarse; pass n_samples for a Monte Carlo estimate from the same Gaussian.
    """
    noise_scale = track.step_dt / horizon
    n_steps = int(round(horizon / dt))
    if paths is not None:
        actor_paths, w = paths
    elif n_samples is None:
        actor_paths, w = predicted_paths(track, horizon, dt, lam)
    else:
        rng = rng or np.random.default_rng()
        x_aug, P_aug = track.augmented(noise_scale)
        pts = rng.multivariate_normal(x_aug, P_aug, size=n_samples)
        w = np.full(n_samples, 1.0 / n_samples)
        actor_paths = rollout(pts, track.agility, dt, n_steps)
    ego_path = ego.rollout(dt, n_steps)
    d_min = np.linalg.norm(actor_paths - ego_path[:, None, :], axis=-1).min(axis=0)
    hit = d_min < combined_radius(track, ego_radius)
    return float(np.clip(w @ hit, 0.0, 1.0))


def predicted_paths(
    track: CTRVUKF,
    horizon: float = 2.5,
    dt: float = 0.1,
    lam: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Augmented sigma points rolled forward over the horizon.

    Returns (paths (n_steps+1, 15, 2) in the world frame, weights (15,)).
    Noise variance is scaled by step_dt / horizon; see conflict_probability.
    This is also what the dashboard draws as the prediction cone.
    """
    if lam < 0:
        raise ValueError("lam must be >= 0 for non-negative weights")
    pts, w = track.augmented_sigma_points(lam, track.step_dt / horizon)
    return rollout(pts, track.agility, dt, int(round(horizon / dt))), w


# ---- Stage E: crawl gate, baseline, alert policy ----------------------------

def crawl_gate(v_ego: float, v_lateral: float,
               ego_speed_max: float = CRAWL_EGO_SPEED,
               lateral_speed_max: float = CRAWL_LATERAL_SPEED) -> bool:
    """True (suppress the alert) when ego < 18 km/h AND lateral closing |v_x| < 1.2 m/s.

    Bumper-to-bumper bunching at crawl speed is normal, not a hazard; an actor
    moving sideways into the ego path faster than 1.2 m/s still gets through.
    """
    return v_ego < ego_speed_max and abs(v_lateral) < lateral_speed_max


def ttc_1d(forward: float, v_forward: float, R: float) -> float:
    """Lane-based 1D TTC: longitudinal gap / closing speed, for actors ahead only."""
    if forward <= 0.0:
        return math.inf
    gap = max(forward - R, 0.0)
    closing = -v_forward
    if closing <= 1e-9:
        return math.inf
    return gap / closing


def side_of(lateral: float, dead_band: float = 0.5) -> str:
    if lateral < -dead_band:
        return "left"
    if lateral > dead_band:
        return "right"
    return "ahead"


@dataclass
class RiskAssessment:
    lateral: float        # m, track position in the ego frame (right +)
    forward: float        # m
    v_lateral: float      # m/s, relative velocity in the ego frame
    v_forward: float
    ttc_2d: float
    p_conf: float
    ttc_1d: float
    converging: bool
    in_range: bool
    crawl: bool
    bharat: bool          # 2D rule, this frame, before debounce
    baseline: bool        # lane-based 1D rule, this frame, before debounce

    @property
    def side(self) -> str:
        return side_of(self.lateral)


def assess(
    track: CTRVUKF,
    ego: EgoState,
    ego_radius: float = DEFAULT_EGO_RADIUS,
    use_crawl_gate: bool = True,
    horizon: float = 2.5,
    paths: tuple[np.ndarray, np.ndarray] | None = None,
) -> RiskAssessment:
    """Score one track against the ego with both the Bharat-TTC rule and the 1D baseline.

    paths: optional precomputed predicted_paths(track, horizon) to reuse.
    """
    from .camera import rotate_to_camera, world_to_camera

    lat, fwd = world_to_camera(track.position, ego)
    v_lat, v_fwd = rotate_to_camera(track.velocity - ego.velocity, ego)
    R = combined_radius(track, ego_radius)
    ttc2 = track_ttc_2d(track, ego, ego_radius)
    p = conflict_probability(track, ego, horizon=horizon, ego_radius=ego_radius, paths=paths)
    ttc1 = ttc_1d(fwd, v_fwd, R)
    converging = lat * v_lat < 0 or abs(lat) < CORRIDOR_HALF_WIDTH
    in_range = math.hypot(lat, fwd) < ALERT_RANGE
    crawl = use_crawl_gate and crawl_gate(ego.v, v_lat)
    bharat = (not crawl) and in_range and p >= P_CONF_MIN and ttc2 <= TTC_2D_MAX and converging
    baseline = abs(lat) < CORRIDOR_HALF_WIDTH and ttc1 <= TTC_1D_MAX
    return RiskAssessment(float(lat), float(fwd), float(v_lat), float(v_fwd), float(ttc2),
                          float(p), float(ttc1), bool(converging), bool(in_range), bool(crawl),
                          bool(bharat), bool(baseline))


@dataclass
class AlertEvent:
    t: float
    actor_id: str
    message: str
    side: str


@dataclass
class AlertPolicy:
    """Debounce (k of n consecutive frames) plus a per-actor voice cooldown.

    update() returns (active, event): active is the debounced alert state for
    this frame; event is set only when a new voice cue should be spoken.
    """
    k: int = 2
    n: int = 3
    cooldown: float = 4.0
    _hist: dict[str, deque] = field(default_factory=dict)
    _last_cue: dict[str, float] = field(default_factory=dict)
    _active: dict[str, bool] = field(default_factory=dict)

    def update(self, t: float, actor_id: str, raw: bool, message: str = "",
               side: str = "") -> tuple[bool, AlertEvent | None]:
        h = self._hist.setdefault(actor_id, deque(maxlen=self.n))
        h.append(bool(raw))
        active = sum(h) >= self.k
        was_active = self._active.get(actor_id, False)
        self._active[actor_id] = active
        event = None
        if active and not was_active:
            last = self._last_cue.get(actor_id, -math.inf)
            if t - last >= self.cooldown:
                self._last_cue[actor_id] = t
                event = AlertEvent(t, actor_id, message, side)
        return active, event
