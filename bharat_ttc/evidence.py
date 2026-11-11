"""Evidence for the deck's hypotheses, computed from the simulator (synthetic only).

  H1  lead time on angular cut-ins: swing angle x noise seed sweep
  H2  nuisance alerts in a crawling jam, crawl gate on vs off
  ablation: class-specific vs shared (truck) agility, for warning time and tracking
  counter-example: merge-then-straighten
  H3 proxy: Layer 1 compute time per actor-frame on this machine (no detector)

Shared by demo_cut_in.py and the dashboard so both report the same numbers.
"""
from __future__ import annotations

import math
import platform
import statistics as stats
import time

import numpy as np

from .camera import CameraModel, calibrate_from_detections, camera_to_world, simulate_calibration_traffic, world_to_camera
from .motion_models import ActorClass, EgoState, ctrv_step, wrap_angle
from .risk import assess, predicted_paths
from .sim import (
    FPS,
    TRUE_CAMERA,
    ablation_truck_agility,
    collision_course_cut_in,
    crawl_jam,
    merge_then_straighten,
    run_scenario,
)
from .ukf import CTRVUKF

SWEEP_ANGLES = (10, 15, 20, 25, 30, 35, 40, 45)
SWEEP_SEEDS = (1, 2, 3, 4, 5)


def _median(xs):
    return stats.median(xs) if xs else None


def sweep(angles=SWEEP_ANGLES, seeds=SWEEP_SEEDS) -> dict:
    rows, all_entry, all_base = [], [], []
    for deg in angles:
        b, l, eg, lg, entry = [], [], [], [], None
        for seed in seeds:
            s = run_scenario(collision_course_cut_in(deg, with_traffic=False, seed=seed), record=False).focus
            entry = s.corridor_entry
            if s.first_bharat is not None:
                b.append(s.first_bharat)
                eg.append(s.corridor_entry - s.first_bharat)
            if s.first_baseline is not None:
                l.append(s.first_baseline)
            if s.lead_gain is not None:
                lg.append(s.lead_gain)
        all_entry += eg
        all_base += lg
        rows.append({
            "angle": deg, "n": len(seeds), "bharat_median": _median(b), "entry": entry,
            "baseline_fired": len(l), "baseline_median": _median(l),
            "lead_vs_entry_median": _median(eg), "lead_vs_entry_min": min(eg) if eg else None,
            "lead_vs_entry_max": max(eg) if eg else None,
            "lead_vs_baseline_median": _median(lg),
        })
    return {
        "rows": rows, "runs": len(angles) * len(seeds),
        "median_lead_vs_entry": _median(all_entry), "median_lead_vs_baseline": _median(all_base),
        "share_over_1s": sum(g > 1.0 for g in all_entry) / max(1, len(all_entry)),
        "t_conflict": collision_course_cut_in().t_conflict,
    }


def crawl_comparison() -> dict:
    out = {}
    for gate in (True, False):
        res = run_scenario(crawl_jam(gate), record=False)
        car, auto = res.summaries["car-1"], res.summaries["auto-2"]
        out["gate_on" if gate else "gate_off"] = {
            "bunching_bharat": car.bharat_cues, "bunching_baseline": car.baseline_cues,
            "squeeze_bharat": auto.bharat_cues, "squeeze_baseline": auto.baseline_cues,
        }
    out["duration_s"] = crawl_jam().duration
    return out


def track_swerve(cls: ActorClass, seed: int, cam: CameraModel = TRUE_CAMERA) -> dict:
    """Auto weaving +-0.6 rad/s (2.5 s period) 12 m ahead of a stopped ego."""
    rng = np.random.default_rng(seed)
    ego = EgoState(0.0, 0.0, 0.0, math.pi / 2)
    dt = 1 / FPS
    x = np.array([-1.0, 12.0, 6.0, math.pi / 2, 0.0])
    trk, pos, hdg = None, [], []
    for k in range(75):
        x[4] = 0.6 * math.sin(2 * math.pi * k * dt / 2.5)
        xp, yb = cam.project(*world_to_camera(x[:2], ego))
        zc, Rc = cam.backproject(xp + rng.normal(0, cam.sigma_x), yb + rng.normal(0, cam.sigma_y))
        z, R = camera_to_world(zc, Rc, ego)
        if trk is None:
            trk = CTRVUKF.from_detection(cls, z, R, heading=math.pi / 2, speed=5.0)
        else:
            trk.predict(dt)
            trk.update(z, R)
            if k > 15:
                pos.append(float(np.linalg.norm(trk.position - x[:2])))
                hdg.append(abs(float(wrap_angle(trk.x[3] - x[3]))))
        x = ctrv_step(x, dt)
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
    return {"pos_rms": rms(pos), "heading_rms": rms(hdg)}


def ablation(seeds=range(10)) -> dict:
    auto = run_scenario(collision_course_cut_in(30.0, with_traffic=False), record=False).focus
    truck = run_scenario(ablation_truck_agility(30.0), record=False).focus
    sw = {}
    for cls in (ActorClass.AUTO_RICKSHAW, ActorClass.TRUCK):
        r = [track_swerve(cls, s) for s in seeds]
        sw[cls.value] = {"pos_rms": float(np.mean([x["pos_rms"] for x in r])),
                         "heading_rms_deg": math.degrees(float(np.mean([x["heading_rms"] for x in r])))}
    merge = run_scenario(merge_then_straighten(), record=False).focus
    return {
        "cut_in": {"auto_prior": auto.first_bharat, "truck_prior": truck.first_bharat,
                   "baseline": auto.first_baseline},
        "swerve_tracking": sw, "swerve_seeds": len(list(seeds)),
        "merge": {"bharat": merge.first_bharat, "baseline": merge.first_baseline, "lead": merge.lead_gain},
    }


def _bench_once(n_frames: int, seed: int) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    ego = EgoState(0.0, 0.0, 13.3, math.pi / 2)
    R = np.diag([0.1**2, 0.4**2])
    x0 = np.array([-4.0, 25.0, 5.5, math.pi / 2 - 0.3, -0.1])
    x = x0.copy()
    trk = CTRVUKF.from_detection(ActorClass.AUTO_RICKSHAW, x[:2], R, heading=math.pi / 2, speed=13.3)
    dt = 1 / FPS
    warmup = 30
    f_ms, r_ms = [], []
    for i in range(n_frames + warmup):
        x = ctrv_step(x, dt)
        z = x[:2] + rng.normal(0, [0.1, 0.4])
        t0 = time.perf_counter()
        trk.predict(dt)
        trk.update(z, R)
        t1 = time.perf_counter()
        assess(trk, ego, paths=predicted_paths(trk))
        t2 = time.perf_counter()
        if i >= warmup:
            f_ms.append((t1 - t0) * 1000)
            r_ms.append((t2 - t1) * 1000)
        if np.linalg.norm(x[:2]) > 60:  # keep the target in a sensible range
            x = x0.copy()
    return f_ms, r_ms


def benchmark(n_frames: int = 200, repeats: int = 5) -> dict:
    """Wall time of Layer 1 per actor-frame: UKF predict+update, then TTC_2D + P_conf rollout.

    Pure Python/NumPy on this machine's CPU; excludes the detector (stage A).
    Not a Jetson measurement. Run as several repeats because on hybrid laptop
    CPUs the OS moves the process between performance and efficiency cores;
    the spread between the best and worst repeat is reported, not hidden.
    """
    f_ms, r_ms, rep_medians = [], [], []
    for k in range(repeats):
        f, r = _bench_once(n_frames, k)
        f_ms += f
        r_ms += r
        rep_medians.append(float(np.median(np.add(f, r))))
    boxes = simulate_calibration_traffic(TRUE_CAMERA, 240, np.random.default_rng(1))
    calibrate_from_detections(boxes)  # warm-up: first call pays sklearn's import
    t0 = time.perf_counter()
    calibrate_from_detections(boxes)
    t_cal = time.perf_counter() - t0
    tot = np.add(f_ms, r_ms)
    per, p95 = float(np.median(tot)), float(np.percentile(tot, 95))
    budget = 1000 / FPS
    return {
        "ms_filter": float(np.median(f_ms)), "ms_risk": float(np.median(r_ms)), "ms_total": per,
        "ms_total_p95": p95, "frames": n_frames * repeats, "repeats": repeats,
        "ms_best_repeat": min(rep_medians), "ms_worst_repeat": max(rep_medians),
        "frame_budget_ms": budget, "actors_in_budget": int(budget // per) if per > 0 else None,
        "actors_in_budget_p95": int(budget // p95) if p95 > 0 else None,
        "calibration_fit_ms": t_cal * 1000, "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def limitations() -> list[str]:
    return [
        "All results are from synthetic scenarios with fixed seeds, not real dashcam footage.",
        "Shallow drifts (10-15°) enter the lane corridor early, so the lead shrinks below 1 s.",
        "A swerve that straightens out ahead is caught later than by the 1D rule (merge case).",
        "On the cut-in, truck agility warns as early as auto agility: the lead comes from 2D geometry; "
        "class agility improves tracking accuracy instead.",
        "The camera cannot see box bottoms within ~3.8 m; tracks coast there for up to 0.5 s.",
        "Agility priors are starting values until estimated on METEOR tracks; detection and ID "
        "association (stage A) are not part of this prototype.",
    ]


def record_sample() -> dict:
    """The near-miss record emitted for the hero alert, with its measured sizes."""
    res = run_scenario(collision_course_cut_in(30.0), record=False)
    ev = next(e for e in res.events if e["system"] == "bharat")
    return {"record": ev["record"], "json_bytes": ev["json_bytes"], "binary_bytes": ev["binary_bytes"],
            "records_per_mb_json": 1_000_000 // ev["json_bytes"],
            "records_per_mb_binary": 1_000_000 // ev["binary_bytes"]}


def compute_all() -> dict:
    return {"sweep": sweep(), "crawl": crawl_comparison(), "ablation": ablation(),
            "benchmark": benchmark(), "record": record_sample(), "limitations": limitations()}
