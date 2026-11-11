"""Deterministic scenario simulator: bounding boxes in, alerts out.

One run exercises the whole Layer 1 pipeline on synthetic data:

  1. Stage B: background traffic boxes (cars, autos, with outliers) seen by
     the *true* camera are fitted with the lane-free calibration; everything
     after uses the *fitted* camera, never the true one.
  2. Every frame, each actor's true ground position is drawn as a bounding
     box [x1, y1, x2, y2] by the true camera, with pixel noise on each edge.
  3. The box's bottom-centre is back-projected with the fitted camera into the
     world frame (with a distance-dependent covariance) and fed to that
     actor's class-specific CTRV UKF.
  4. Each track is scored by the Bharat-TTC 2D rule (P_conf, TTC_2D,
     convergence, crawl gate) and by the lane-based 1D baseline (1D TTC < 2 s
     in a fixed 1.5 m half-width corridor). Both go through the same 2-of-3
     debounce, so the only difference is the risk logic.

Association (stage A) is out of scope: detections carry their true actor id.
World frame: the ego starts at the origin heading +Y (psi = pi/2), so world
X is the ego's lateral axis (right positive).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from .camera import (
    VEHICLE_SIZE,
    CameraModel,
    calibrate_from_detections,
    camera_to_world,
    simulate_calibration_traffic,
    world_to_camera,
)
from .motion_models import ActorClass, EgoState, ctrv_step, wrap_angle
from .risk import (
    CORRIDOR_HALF_WIDTH,
    AlertPolicy,
    assess,
    combined_radius,
    predicted_paths,
)
from .records import SCHEMA_VERSION, NearMissRecord, drac, h3_cell, local_to_latlon
from .ukf import CTRVUKF

FPS = 15.0
DT = 1.0 / FPS
IMAGE_W, IMAGE_H = 1280, 720
TRUE_CAMERA = CameraModel(f=700.0, cx=640.0, y_h=330.0, h=2.1)  # wide-angle truck dashcam, ~85 deg HFOV
DETECT_RANGE = 60.0  # m; beyond this the detector is assumed to miss small actors
MAX_COAST_FRAMES = 8  # ~0.5 s of prediction-only tracking before a lost track is dropped
# Simulated GPS origin and clock for near-miss records (synthetic, not a real trip).
SIM_ORIGIN = (28.4595, 77.0266)   # Delhi NCR, illustrative
SIM_EPOCH = 1793334600            # fixed start time so records are deterministic

SpeedFn = Callable[[float], float]


@dataclass
class ActorSpec:
    """An actor spawned at spawn_t, placed relative to the ego at that moment.

    heading_deg is relative to the ego heading, anticlockwise positive, so an
    actor on the left turning in toward the ego path has a *negative* yaw rate.
    yaw_schedule entries are (t_start, t_end, yaw_rate rad/s) in sim time.
    speed_fn, if given, overrides the speed every frame (stop-and-go traffic).
    """
    actor_id: str
    true_class: ActorClass
    lateral: float
    forward: float
    speed: float
    heading_deg: float = 0.0
    yaw_schedule: list[tuple[float, float, float]] = field(default_factory=list)
    spawn_t: float = 0.0
    speed_fn: SpeedFn | None = None
    tracked_as: ActorClass | None = None

    @property
    def track_class(self) -> ActorClass:
        return self.tracked_as or self.true_class


@dataclass
class ScenarioSpec:
    key: str
    title: str
    description: str
    ego_speed_kmh: float
    actors: list[ActorSpec]
    duration: float = 6.0
    ego_speed_fn: SpeedFn | None = None
    use_crawl_gate: bool = True
    focus: str | None = None  # actor id the head-to-head comparison is about
    seed: int = 7
    t_conflict: float | None = None  # when the focus actor would hit the ego if nobody braked


# ---- scenario library --------------------------------------------------------

def collision_course_cut_in(
    swing_deg: float = 30.0,
    ego_kmh: float = 48.0,
    auto_kmh: float = 20.0,
    swing_start: float = 1.2,
    swing_time: float = 0.9,
    t_conflict: float = 3.6,
    aim_lateral: float = 0.0,
    tracked_as: ActorClass | None = None,
    with_traffic: bool = True,
    use_crawl_gate: bool = True,
    seed: int = 7,
    key: str = "hero",
    title: str | None = None,
) -> ScenarioSpec:
    """Auto rides parallel ahead-left, then swings swing_deg to the right and holds the diagonal.

    The start position is solved backwards so that, if nobody brakes, the auto
    is at aim_lateral (m) on the truck's centreline at t_conflict: every swing
    angle is a genuine collision course, so lead times are comparable.
    """
    v_e, v_a = ego_kmh / 3.6, auto_kmh / 3.6
    rate = -math.radians(swing_deg) / swing_time
    sched = [(swing_start, swing_start + swing_time, rate)]
    x = np.array([0.0, 0.0, v_a, math.pi / 2, 0.0])
    for k in range(int(round(t_conflict / DT))):
        x[4] = next((w for t0, t1, w in sched if t0 <= k * DT < t1), 0.0)
        x = ctrv_step(x, DT)
    lateral, forward = aim_lateral - x[0], v_e * t_conflict - x[1]
    actors = [ActorSpec("auto-1", ActorClass.AUTO_RICKSHAW, lateral, forward, v_a, 0.0, sched,
                        tracked_as=tracked_as)]
    if with_traffic:
        actors += [
            # Rider holding a parallel line on the ego's right: TTC_2D must stay infinite.
            ActorSpec("bike-1", ActorClass.TWO_WHEELER, 3.2, 6.0, v_e + 0.4),
            # Oncoming truck in the opposite flow on an undivided road.
            ActorSpec("truck-1", ActorClass.TRUCK, 7.0, 58.0, 40 / 3.6, 180.0),
        ]
    return ScenarioSpec(
        key, title or f"Auto-rickshaw {swing_deg:.0f}° diagonal cut-in, truck at {ego_kmh:.0f} km/h",
        f"Ego truck cruises at {ego_kmh:.0f} km/h. An auto ahead-left at {auto_kmh:.0f} km/h "
        f"swings {swing_deg:.0f}° across the truck's path toward a gap, on a collision course "
        f"(conflict at t = {t_conflict:.1f} s if nobody brakes). A parallel rider and an "
        "oncoming truck should stay quiet.",
        ego_kmh, actors, duration=t_conflict + 0.6, use_crawl_gate=use_crawl_gate,
        focus="auto-1", seed=seed, t_conflict=t_conflict,
    )


def ablation_truck_agility(swing_deg: float = 30.0) -> ScenarioSpec:
    sc = collision_course_cut_in(swing_deg, tracked_as=ActorClass.TRUCK, with_traffic=False,
                                 key="ablation",
                                 title="Ablation: same cut-in, auto tracked with truck agility")
    sc.description = ("Identical detections, but the auto is tracked with bus/truck agility "
                      "(sigma_a 1.2, sigma_psidd 0.25): narrower cone, slower to believe the turn.")
    return sc


def merge_then_straighten() -> ScenarioSpec:
    """Honest counter-example: the swerve straightens out, so 2D has no edge."""
    return ScenarioSpec(
        "merge", "Auto merges, then straightens 7 m ahead (30 km/h)",
        "The auto swerves into the ego path and straightens out ahead of the faster "
        "truck. Constant-turn prediction first reads the swerve as a crossing that "
        "clears, so here the lane-based check is not beaten.",
        30.0,
        [ActorSpec("auto-1", ActorClass.AUTO_RICKSHAW, -4.0, 20.0, 5.0, 0.0,
                   [(1.5, 2.7, -0.6), (2.7, 3.9, 0.6)])],
        duration=6.0, focus="auto-1",
    )


def _stop_and_go(mean: float, amp: float, period: float, phase: float) -> SpeedFn:
    return lambda t: max(0.0, mean + amp * math.sin(2 * math.pi * (t - phase) / period))


def crawl_jam(use_crawl_gate: bool = True) -> ScenarioSpec:
    """Stop-and-go jam: the 1D baseline nags on every bunching; the gate keeps quiet.

    From t = 7.4 s an auto squeezes in from the left at > 1.2 m/s lateral speed,
    which the crawl gate must *not* suppress.
    """
    period = 4.0
    return ScenarioSpec(
        "jam" if use_crawl_gate else "jam_nogate",
        "Crawling jam, stop-and-go under 15 km/h" + ("" if use_crawl_gate else " (crawl gate OFF)"),
        "Bumper-to-bumper bunching behind a car is normal at crawl speed. The 1D "
        "baseline fires on every squeeze; the crawl gate suppresses those. An auto "
        "squeezing in sideways later is a real hazard and still gets a cue.",
        8.0,
        [
            ActorSpec("car-1", ActorClass.CAR, 0.0, 6.5, 2.2,
                      speed_fn=_stop_and_go(2.2, 2.0, period, -0.9)),
            ActorSpec("auto-2", ActorClass.AUTO_RICKSHAW, -4.6, 4.0, 2.2, 0.0,
                      [(7.4, 8.2, -0.75)], spawn_t=6.0, speed_fn=lambda t: 2.6),
        ],
        duration=12.0, ego_speed_fn=_stop_and_go(2.2, 2.0, period, 0.0),
        use_crawl_gate=use_crawl_gate, focus="car-1",
    )


def scenario_library() -> dict[str, ScenarioSpec]:
    scs = [collision_course_cut_in(), ablation_truck_agility(), merge_then_straighten(),
           crawl_jam(True), crawl_jam(False)]
    return {s.key: s for s in scs}


# ---- simulation --------------------------------------------------------------

@dataclass
class ActorSummary:
    actor_id: str
    true_class: str
    tracked_as: str
    corridor_entry: float | None = None   # first time the TRUE position is inside the corridor
    first_bharat: float | None = None     # first debounced 2D alert
    first_baseline: float | None = None   # first debounced 1D alert
    bharat_cues: int = 0                  # voice cues spoken
    baseline_cues: int = 0                # baseline alert onsets (beeps)
    min_ttc_2d: float = math.inf
    min_distance: float = math.inf        # true closest approach (m), centre to centre

    @property
    def lead_gain(self) -> float | None:
        if self.first_bharat is None or self.first_baseline is None:
            return None
        return self.first_baseline - self.first_bharat


@dataclass
class SimResult:
    spec: ScenarioSpec
    calibration: dict
    frames: list[dict]
    summaries: dict[str, ActorSummary]
    events: list[dict]

    @property
    def focus(self) -> ActorSummary | None:
        return self.summaries.get(self.spec.focus) if self.spec.focus else None


def _simulate_ego(spec: ScenarioSpec, n: int) -> list[EgoState]:
    s = np.array([0.0, 0.0, spec.ego_speed_kmh / 3.6, math.pi / 2, 0.0])
    egos = []
    for k in range(n):
        if spec.ego_speed_fn is not None:
            s[2] = spec.ego_speed_fn(k * DT)
        egos.append(EgoState(*s))
        s = ctrv_step(s, DT)
    return egos


def _simulate_actor(a: ActorSpec, egos: list[EgoState], n: int) -> list[np.ndarray | None]:
    k0 = int(round(a.spawn_t / DT))
    out: list[np.ndarray | None] = [None] * n
    if k0 >= n:
        return out
    e = egos[k0]
    p = camera_to_world(np.array([a.lateral, a.forward]), np.zeros((2, 2)), e)[0]
    x = np.array([p[0], p[1], a.speed, wrap_angle(e.psi + math.radians(a.heading_deg)), 0.0])
    for k in range(k0, n):
        t = k * DT
        x[4] = next((w for t0, t1, w in a.yaw_schedule if t0 <= t < t1), 0.0)
        if a.speed_fn is not None:
            x[2] = a.speed_fn(t)
        out[k] = x.copy()
        x = ctrv_step(x, DT)
    return out


def _detect(cam_true: CameraModel, cls: ActorClass, lat: float, fwd: float,
            rng: np.random.Generator) -> np.ndarray | None:
    if fwd < 1.5 or fwd > DETECT_RANGE:
        return None
    W, Hh = VEHICLE_SIZE[cls]
    box = cam_true.bbox(lat, fwd, W, Hh)
    # Edge noise chosen so the bottom-centre has std sigma_x, sigma_y.
    box += rng.normal(0.0, 1.0, 4) * np.array([cam_true.sigma_x * math.sqrt(2), cam_true.sigma_y,
                                                 cam_true.sigma_x * math.sqrt(2), cam_true.sigma_y])
    xc = 0.5 * (box[0] + box[2])
    # The bottom edge must be in the image or y_b (hence range) is unusable:
    # with this mount that is a near blind zone of ~3.8 m ahead of the camera.
    if not (0.0 <= xc <= IMAGE_W) or box[3] > IMAGE_H:
        return None
    return box


def _near_miss_record(spec: ScenarioSpec, egos: list[EgoState], k: int, a: ActorSpec,
                      ra, min_ttc: float) -> NearMissRecord:
    """The ~300-byte Layer 2 record the device would upload for this alert."""
    ego = egos[k]
    lat, lon = local_to_latlon(ego.x, ego.y, *SIM_ORIGIN)
    accel = (ego.v - egos[k - 1].v) / DT if k > 0 else 0.0
    r = np.array([ra.lateral, ra.forward]); v = np.array([ra.v_lateral, ra.v_forward])
    closing = float(-(r @ v) / max(np.linalg.norm(r), 1e-6))
    return NearMissRecord(
        v=SCHEMA_VERSION, h3=h3_cell(lat, lon), t=int(SIM_EPOCH + k * DT), ego_kmh=ego.v * 3.6,
        min_ttc_s=min(min_ttc, 99.0), ego_accel=accel, drac=drac(closing, ra.ttc_2d),
        p_conf=ra.p_conf, actor=a.true_class.value, side=ra.side, light="day", dev="b7c41e",
    )


def _ellipse(P2: np.ndarray) -> list[float]:
    vals, vecs = np.linalg.eigh(0.5 * (P2 + P2.T))
    vals = np.clip(vals, 0.0, None)
    ang = math.atan2(vecs[1, 1], vecs[0, 1])
    return [round(2 * math.sqrt(vals[1]), 3), round(2 * math.sqrt(vals[0]), 3), round(ang, 3)]


def _r(v: float, nd: int = 3):
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


def run_scenario(
    spec: ScenarioSpec,
    cam_true: CameraModel = TRUE_CAMERA,
    record: bool = True,
    path_stride: int = 5,
) -> SimResult:
    rng = np.random.default_rng(spec.seed)

    # Stage B: calibrate from traffic, then use only the fitted camera.
    calib_boxes = simulate_calibration_traffic(cam_true, 240, rng, image_height=IMAGE_H)
    cal = calibrate_from_detections(calib_boxes, method="huber")
    cam = CameraModel.from_calibration(cal, f=cam_true.f, cx=cam_true.cx,
                                       sigma_x=cam_true.sigma_x, sigma_y=cam_true.sigma_y)
    calibration = {
        "method": cal.method, "n": cal.n, "inlier_fraction": round(cal.inlier_fraction, 3),
        "h_fit": round(cal.h, 4), "h_true": cam_true.h,
        "y_h_fit": round(cal.y_h, 2), "y_h_true": cam_true.y_h,
        "samples": [[round(float(b[3]), 1), round(float(b[2] - b[0]), 2), c.value]
                    for c, b in calib_boxes] if record else [],
    }

    n = int(round(spec.duration * FPS))
    egos = _simulate_ego(spec, n)
    truths = {a.actor_id: _simulate_actor(a, egos, n) for a in spec.actors}
    tracks: dict[str, CTRVUKF] = {}
    hits: dict[str, int] = {}     # detections absorbed by each track
    misses: dict[str, int] = {}   # consecutive frames coasted without a detection
    bharat_policy, baseline_policy = AlertPolicy(), AlertPolicy()
    summaries = {a.actor_id: ActorSummary(a.actor_id, a.true_class.value, a.track_class.value)
                 for a in spec.actors}
    frames: list[dict] = []
    events: list[dict] = []

    for k in range(n):
        t = round(k * DT, 4)
        ego = egos[k]
        frame_actors = []
        for a in spec.actors:
            truth = truths[a.actor_id][k]
            if truth is None:
                continue
            summ = summaries[a.actor_id]
            lat_t, fwd_t = world_to_camera(truth[:2], ego)
            summ.min_distance = min(summ.min_distance, math.hypot(lat_t, fwd_t))
            if summ.corridor_entry is None and abs(lat_t) < CORRIDOR_HALF_WIDTH and fwd_t > 0:
                summ.corridor_entry = t

            box = _detect(cam_true, a.true_class, lat_t, fwd_t, rng)
            trk = tracks.get(a.actor_id)
            nis = None
            if box is not None:
                try:
                    z_cam, R_cam = cam.backproject_bbox(box)
                except ValueError:
                    box = None
            if box is not None:
                z, Rw = camera_to_world(z_cam, R_cam, ego)
                if trk is None:
                    trk = CTRVUKF.from_detection(a.track_class, z, Rw, heading=ego.psi, speed=ego.v)
                    tracks[a.actor_id] = trk
                    hits[a.actor_id] = 1
                else:
                    trk.predict(DT)
                    nis = trk.update(z, Rw)
                    hits[a.actor_id] += 1
                misses[a.actor_id] = 0
            elif trk is not None:
                # Coast through missed detections (e.g. the near blind zone), then drop.
                misses[a.actor_id] += 1
                if misses[a.actor_id] > MAX_COAST_FRAMES:
                    del tracks[a.actor_id]
                    trk = None
                else:
                    trk.predict(DT)

            entry: dict = {
                "id": a.actor_id, "cls": a.true_class.value, "tracked_as": a.track_class.value,
                "truth": [_r(lat_t), _r(fwd_t)],
                "bbox": None if box is None else [round(float(v), 1) for v in box],
                "coasting": trk is not None and box is None,
            }
            if trk is not None and hits.get(a.actor_id, 0) >= 2:  # need >= 2 detections to score
                pw = predicted_paths(trk)  # one rollout, used for P_conf and for drawing
                ra = assess(trk, ego, use_crawl_gate=spec.use_crawl_gate, paths=pw)
                label = f"{a.true_class.spoken}, {ra.side}!"
                b_on, b_ev = bharat_policy.update(t, a.actor_id, ra.bharat, label, ra.side)
                l_on, l_ev = baseline_policy.update(t, a.actor_id, ra.baseline, "Brake!", ra.side)
                summ.min_ttc_2d = min(summ.min_ttc_2d, ra.ttc_2d)
                if b_on and summ.first_bharat is None:
                    summ.first_bharat = t
                if l_on and summ.first_baseline is None:
                    summ.first_baseline = t
                if b_ev:
                    summ.bharat_cues += 1
                    rec = _near_miss_record(spec, egos, k, a, ra, summ.min_ttc_2d)
                    events.append({"t": t, "system": "bharat", "id": a.actor_id,
                                   "message": b_ev.message, "side": b_ev.side,
                                   "record": rec.to_json(), **rec.sizes()})
                if l_ev:
                    summ.baseline_cues += 1
                    events.append({"t": t, "system": "baseline", "id": a.actor_id,
                                   "message": l_ev.message, "side": l_ev.side})
                if record:
                    paths, w = pw
                    # World -> ego frame rotation (also for the covariance ellipse).
                    M = np.array([[math.sin(ego.psi), -math.cos(ego.psi)],
                                  [math.cos(ego.psi), math.sin(ego.psi)]])
                    # Futures relative to the ego's own future, p_i(t) - p_ego(t), in the
                    # current ego frame: a conflict is a cone that reaches the ego disc.
                    ego_path = ego.rollout(0.1, paths.shape[0] - 1)
                    sub = (paths - ego_path[:, None, :])[::path_stride] @ M.T  # (steps, 15, 2)
                    rel = np.round(np.transpose(sub, (1, 0, 2)), 2).tolist()
                    entry.update({
                        "track": {
                            "pos": [_r(ra.lateral), _r(ra.forward)],
                            "vel": [_r(ra.v_lateral), _r(ra.v_forward)],
                            "speed": _r(trk.x[2]),
                            "heading_rel": _r(wrap_angle(trk.x[3] - ego.psi)),
                            "yaw_rate": _r(trk.x[4]),
                            "ellipse": _ellipse(M @ trk.P[:2, :2] @ M.T),
                            "nis": _r(nis),
                        },
                        "paths": rel,
                        "weights": [round(float(v), 4) for v in w],
                        "risk": {
                            "ttc_2d": _r(ra.ttc_2d), "ttc_1d": _r(ra.ttc_1d),
                            "p_conf": _r(ra.p_conf), "converging": ra.converging,
                            "crawl": ra.crawl, "bharat_raw": ra.bharat,
                            "baseline_raw": ra.baseline, "R": _r(combined_radius(trk)),
                        },
                        "bharat": b_on, "baseline": l_on,
                    })
            frame_actors.append(entry)

        if record:
            frames.append({
                "t": t,
                "ego": {"v": _r(ego.v), "kmh": _r(ego.v * 3.6, 1)},
                "ego_path": [[0.0, _r(ego.v * s * 0.5, 2)] for s in range(6)],
                "actors": frame_actors,
            })

    return SimResult(spec, calibration, frames, summaries, events)


def result_to_json(res: SimResult) -> dict:
    sc = res.spec
    return {
        "scenario": {
            "key": sc.key, "title": sc.title, "description": sc.description,
            "ego_speed_kmh": sc.ego_speed_kmh, "duration": sc.duration, "fps": FPS,
            "use_crawl_gate": sc.use_crawl_gate, "focus": sc.focus, "t_conflict": sc.t_conflict,
            "image": [IMAGE_W, IMAGE_H],
            "camera": {"f": TRUE_CAMERA.f, "cx": TRUE_CAMERA.cx,
                       "y_h": res.calibration["y_h_fit"], "h": res.calibration["h_fit"]},
            "corridor_half_width": CORRIDOR_HALF_WIDTH,
        },
        "calibration": res.calibration,
        "summaries": {k: {**vars(s), "lead_gain": s.lead_gain,
                          "min_ttc_2d": _r(s.min_ttc_2d), "min_distance": _r(s.min_distance)}
                      for k, s in res.summaries.items()},
        "events": res.events,
        "frames": res.frames,
    }


def with_overrides(spec: ScenarioSpec, **kw) -> ScenarioSpec:
    return replace(spec, **kw)
