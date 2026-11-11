"""Deterministic Layer 1 demo: an auto-rickshaw cuts in diagonally across an unlaned road.

Ego truck at 48 km/h. Every frame (15 fps) the auto's true position is drawn as
a dashcam bounding box [x1, y1, x2, y2] with pixel noise. The boxes go through
the whole bharat_ttc pipeline:

  B  lane-free calibration (camera height + horizon from car/auto widths, Huber fit)
  C  bottom-centre back-projection -> class-specific CTRV UKF (filterpy)
  D  TTC_2D + sigma-point P_conf over 2.5 s
  E  crawl gate, 2-of-3 debounce, side-naming voice cue

and the same tracks are scored by the lane-based 1D baseline (1D TTC < 2 s
inside a fixed 1.5 m half-width corridor) with the same debounce. The timeline
prints when each one fires.

All numbers are from synthetic scenarios with fixed seeds, not real footage.

    cd Bharat-TTC
    ..\\.venv\\Scripts\\python.exe demo_cut_in.py
"""
from __future__ import annotations

import math
import sys

import numpy as np

from bharat_ttc import evidence
from bharat_ttc import CLASS_AGILITY, ActorClass, calibrate_from_detections, simulate_calibration_traffic
from bharat_ttc.sim import (
    DT,
    TRUE_CAMERA,
    ablation_truck_agility,
    collision_course_cut_in,
    crawl_jam,
    merge_then_straighten,
    run_scenario,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RULE = "=" * 96


def fmt_t(t: float | None) -> str:
    return "never" if t is None else f"{t:5.2f} s"


def fmt_v(v, nd=2, width=6) -> str:
    return f"{'inf':>{width}}" if v is None else f"{v:{width}.{nd}f}"


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def calibration_report() -> None:
    section("STAGE B  Lane-free calibration from traffic (no lane markings)")
    dets = simulate_calibration_traffic(TRUE_CAMERA, 240, np.random.default_rng(7), image_height=720)
    n_car = sum(c is ActorClass.CAR for c, _ in dets)
    print(f"{len(dets)} background boxes: {n_car} cars (W_c = 1.75 m), {len(dets) - n_car} autos "
          f"(W_c = 1.35 m), ~15% contaminated (mislabelled trucks, occlusions)")
    print("fit  w/W_c = (1/h) * y_b - y_h/h\n")
    print(f"  {'method':8s} {'h (m)':>8s} {'y_h (px)':>9s} {'inliers':>8s}")
    print(f"  {'truth':8s} {TRUE_CAMERA.h:8.3f} {TRUE_CAMERA.y_h:9.1f}")
    for m in ("huber", "ransac"):
        cal = calibrate_from_detections(dets, method=m)
        print(f"  {m:8s} {cal.h:8.3f} {cal.y_h:9.1f} {cal.inlier_fraction:8.0%}")
    print("\nThe pipeline below uses the Huber-fitted camera, never the true one.")


def hero_timeline() -> None:
    spec = collision_course_cut_in(30.0)
    res = run_scenario(spec)
    s = res.focus
    swing = spec.actors[0].yaw_schedule[0]

    section(f"HERO  {spec.title}")
    print(spec.description)
    print(f"Auto starts {abs(spec.actors[0].lateral):.1f} m left, {spec.actors[0].forward:.1f} m ahead; "
          f"swings {swing[0]:.1f}-{swing[1]:.1f} s; conflict at t = {spec.t_conflict:.1f} s if nobody brakes.")
    print(f"Calibration used: h = {res.calibration['h_fit']:.3f} m, y_h = {res.calibration['y_h_fit']:.1f} px\n")

    print("   t    | auto bbox x1,y1,x2,y2 (px)  | track lat  fwd | TTC_2D P_conf TTC_1D |  2D rule   1D rule")
    print("-" * 96)
    ev_by_t = {}
    for e in res.events:
        ev_by_t.setdefault(e["t"], []).append(e)
    for k, f in enumerate(res.frames):
        a = next((x for x in f["actors"] if x["id"] == "auto-1"), None)
        if a is None:
            continue
        t = f["t"]
        events_here = ev_by_t.get(t, [])
        dense = swing[0] - 0.25 <= t <= spec.t_conflict  # every frame around the cut-in
        if not (dense or k % 3 == 0 or events_here):
            continue
        if a["bbox"] is not None:
            box = "{:6.0f},{:4.0f},{:6.0f},{:4.0f}    ".format(*a["bbox"])
        elif a["coasting"]:
            box = "   (blind zone, coasting)   "
        else:
            box = "   (track dropped)          "
        if "risk" in a:
            r = a["risk"]
            lat, fwd = a["track"]["pos"]
            b = "ALERT" if a["bharat"] else ("raw" if r["bharat_raw"] else ".")
            l = "ALERT" if a["baseline"] else ("raw" if r["baseline_raw"] else ".")
            print(f" {t:5.2f}  | {box}| {lat:5.1f} {fwd:5.1f}   | {fmt_v(r['ttc_2d'])} {r['p_conf']:6.2f} "
                  f"{fmt_v(r['ttc_1d'])} | {b:>7s}   {l:>7s}")
        else:
            print(f" {t:5.2f}  | {box}| {'(track starting)' if a['bbox'] else '':16s}|")
        for e in events_here:
            if e["id"] != "auto-1":
                continue
            who = "BHARAT-TTC voice cue" if e["system"] == "bharat" else "1D BASELINE beep"
            print(f"   >>> {t:5.2f} s  {who}: \"{e['message']}\"")

    print("\nHEAD-TO-HEAD (same detections, tracker and calibration; only the risk logic differs)")
    print(f"  Bharat-TTC 2D engine fires : {fmt_t(s.first_bharat)}   "
          f"({spec.t_conflict - s.first_bharat:.2f} s before conflict)")
    if s.first_baseline is None:
        print(f"  1D lane-based baseline     : never   (auto entered the corridor at {s.corridor_entry:.2f} s, "
              f"only {spec.t_conflict - s.corridor_entry:.2f} s before conflict, at the camera's near blind zone)")
    else:
        print(f"  1D lane-based baseline     : {fmt_t(s.first_baseline)}   "
              f"({spec.t_conflict - s.first_baseline:.2f} s before conflict)")
    entry_gain = s.corridor_entry - s.first_bharat
    print(f"  Lead vs corridor entry     : +{entry_gain:.2f} s   (lower bound: no corridor rule can fire "
          "before the auto is in the corridor)")
    if s.lead_gain is not None:
        print(f"  Lead vs tracked baseline   : +{s.lead_gain:.2f} s")
    print(f"  At {spec.ego_speed_kmh:.0f} km/h that is ~{entry_gain * spec.ego_speed_kmh / 3.6:.0f} m more room "
          "to brake.")
    print("\n  Other actors in the scene:")
    for aid in ("bike-1", "truck-1"):
        o = res.summaries[aid]
        print(f"    {aid:8s} ({o.true_class:11s}) min TTC_2D {fmt_v(o.min_ttc_2d, 1)}, "
              f"closest {o.min_distance:4.1f} m, alerts: 2D {o.bharat_cues}, 1D {o.baseline_cues}")


def sweep() -> None:
    section("SWEEP  Collision-course cut-ins, swing angle x 5 noise seeds (truck 48 km/h, auto 20 km/h)")
    print(" angle | 2D fires (median) | corridor entry | 1D fires (n/5, median) | lead vs entry | lead vs 1D")
    print("-" * 96)
    sw = evidence.sweep()
    med = lambda v: "  n/a  " if v is None else f"{v:5.2f} s"  # noqa: E731
    for r in sw["rows"]:
        print(f"  {r['angle']:3d}° |      {med(r['bharat_median'])}      |    {r['entry']:5.2f} s    |      "
              f"{r['baseline_fired']}/{r['n']}  {med(r['baseline_median'])}       |   "
              f"{med(r['lead_vs_entry_median'])}    |  {med(r['lead_vs_baseline_median'])}")
    print()
    print(f"  Median lead vs corridor entry over all {sw['runs']} runs: +{sw['median_lead_vs_entry']:.2f} s"
          f"   ({sw['share_over_1s']:.0%} of runs above 1.0 s; deck target H1: median gain >= 1.0 s)")
    print("  Shallow drifts (10-15°) enter the corridor early, so the 2D edge shrinks below 1 s there.")


def layer2_and_runtime() -> None:
    section("LAYER 2 RECORD AND RUNTIME")
    res = run_scenario(collision_course_cut_in(30.0), record=False)
    ev = next(e for e in res.events if e["system"] == "bharat")
    print("  Near-miss record the device uploads for the hero alert (no video, no driver data):")
    print(f"    {ev['record']}")
    print(f"    size: {ev['json_bytes']} bytes as JSON, {ev['binary_bytes']} bytes packed   (deck: ~300 bytes)")
    print(f"    the deck's < 1 MB/day budget (H5) allows ~{1_000_000 // ev['json_bytes']:,} JSON records per vehicle per day")
    b = evidence.benchmark()
    print()
    print(f"  Layer 1 compute per tracked actor per frame: median {b['ms_total']:.2f} ms, "
          f"p95 {b['ms_total_p95']:.2f} ms  (UKF {b['ms_filter']:.2f} + TTC_2D/P_conf {b['ms_risk']:.2f})")
    print(f"    -> ~{b['actors_in_budget']} actors (p95: {b['actors_in_budget_p95']}) inside one 15 fps frame "
          f"({b['frame_budget_ms']:.0f} ms); calibration refit {b['calibration_fit_ms']:.0f} ms")
    print("    measured on this laptop CPU in pure Python, detector excluded; not a Jetson figure (H3 open).")


def other_scenarios() -> None:
    section("ABLATIONS AND COUNTER-EXAMPLES")
    rows = [
        ("Class-specific agility (auto prior)", collision_course_cut_in(30.0, with_traffic=False)),
        ("Shared agility: auto tracked as truck", ablation_truck_agility(30.0)),
        ("Merge, then straighten (30 km/h)", merge_then_straighten()),
    ]
    print(f"  {'scenario':40s} {'2D fires':>9s} {'1D fires':>9s} {'entry':>8s}  note")
    for name, spec in rows:
        s = run_scenario(spec, record=False).focus
        note = ""
        if "Merge" in name:
            note = f"1D first by {-s.lead_gain:.2f} s: the swerve reads as a crossing that clears"
        print(f"  {name:40s} {fmt_t(s.first_bharat):>9s} {fmt_t(s.first_baseline):>9s} "
              f"{fmt_t(s.corridor_entry):>8s}  {note}")
    print("  Note: on this cut-in the truck prior warns at about the same time. The lead comes from")
    print("  2D geometry; the class prior matters for tracking (swerve position RMS 0.27 m auto prior")
    print("  vs 0.68 m truck prior; see tests/test_layer1.py::test_auto_agility_tracks_swerve...).")

    print("\n  Crawling jam, 12 s stop-and-go under 15 km/h (voice cues / beeps):")
    for gate in (True, False):
        res = run_scenario(crawl_jam(gate), record=False)
        car, auto = res.summaries["car-1"], res.summaries["auto-2"]
        print(f"    crawl gate {'ON ' if gate else 'OFF'}: car ahead bunching -> 2D {car.bharat_cues}, "
              f"1D {car.baseline_cues}   |   auto squeezing in sideways -> 2D {auto.bharat_cues}, "
              f"1D {auto.baseline_cues}")
    print("  The gate removes the bunching nags and still lets the sideways squeeze-in through.")


def main() -> None:
    print("Bharat-TTC Layer 1 - deterministic synthetic demo (fixed seeds; not real footage)")
    calibration_report()
    hero_timeline()
    sweep()
    other_scenarios()
    layer2_and_runtime()
    print("\nAgility priors used (sigma_a m/s^2, sigma_psidd rad/s^2):")
    for c in (ActorClass.AUTO_RICKSHAW, ActorClass.TWO_WHEELER, ActorClass.BUS, ActorClass.TRUCK):
        a = CLASS_AGILITY[c]
        print(f"  {c.value:14s} {a.sigma_a:4.1f}  {a.sigma_psidd:4.2f}")


if __name__ == "__main__":
    main()
