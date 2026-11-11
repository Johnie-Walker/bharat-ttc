import math

import numpy as np
import pytest
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from bharat_ttc import (
    CALIBRATION_WIDTH,
    CLASS_AGILITY,
    CTRVUKF,
    ActorClass,
    AlertPolicy,
    CameraModel,
    EgoState,
    assess,
    calibrate_from_detections,
    calibrate_from_vehicles,
    camera_to_world,
    conflict_probability,
    crawl_gate,
    ctrv_step,
    estimate_agility,
    process_noise,
    rollout,
    simulate_calibration_traffic,
    track_ttc_2d,
    ttc_1d,
    ttc_2d,
    world_to_camera,
    wrap_angle,
)
from bharat_ttc.sim import (
    collision_course_cut_in,
    crawl_jam,
    merge_then_straighten,
    run_scenario,
)

CAM = CameraModel(f=1000.0, cx=640.0, y_h=360.0, h=2.0)
DASHCAM = CameraModel(f=700.0, cx=640.0, y_h=330.0, h=2.1)  # wide-angle truck dashcam


# ---- TTC_2D closed form ----------------------------------------------------

def test_ttc_head_on():
    assert ttc_2d([0, 20], [0, -10], 2.0) == pytest.approx(1.8)


def test_ttc_tangent_graze():
    # Closest approach exactly R: discriminant is zero.
    assert ttc_2d([2, 10], [0, -5], 2.0) == pytest.approx(2.0)


@pytest.mark.parametrize("r, v", [
    ([0, 20], [0, 10]),    # diverging
    ([10, 10], [-10, 0]),  # crossing well clear
    ([0, 20], [0, 0]),     # no relative motion
])
def test_ttc_no_conflict(r, v):
    assert ttc_2d(r, v, 2.0) == math.inf


def test_ttc_already_within_radius():
    assert ttc_2d([0.5, 1.0], [0, 3], 2.0) == 0.0


def test_ttc_matches_brute_force():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 10, 100001)
    for _ in range(50):
        r, v, R = rng.uniform(-20, 20, 2), rng.uniform(-10, 10, 2), rng.uniform(1, 4)
        if r @ r <= R * R:
            continue
        d = np.linalg.norm(r[None] + t[:, None] * v[None], axis=1)
        inside = np.nonzero(d < R)[0]
        expected = t[inside[0]] if inside.size else math.inf
        got = ttc_2d(r, v, R)
        if expected == math.inf:
            assert got == math.inf or got > 10
        else:
            assert got == pytest.approx(expected, abs=1e-3)


# ---- CTRV model --------------------------------------------------------------

def test_ctrv_straight_line():
    x = ctrv_step([0, 0, 5, 0, 0], 2.0)
    assert x[:2] == pytest.approx([10, 0])


def test_ctrv_full_circle_returns_to_start():
    w, n = 0.5, 1000
    x = np.array([3.0, -1.0, 4.0, 0.3, w])
    for _ in range(n):
        x = ctrv_step(x, 2 * math.pi / w / n)
    assert x[:2] == pytest.approx([3.0, -1.0], abs=1e-2)


def test_ctrv_vectorised_matches_scalar():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 5))
    nu = rng.normal(size=(8, 2))
    batch = ctrv_step(X, 0.1, nu)
    for i in range(8):
        assert batch[i] == pytest.approx(ctrv_step(X[i], 0.1, nu[i]))


# ---- Camera measurement model ------------------------------------------------

def test_backproject_round_trip_and_range_noise():
    for X, Z in [(-3.0, 8.0), (1.2, 20.0), (0.0, 35.0)]:
        z, R = CAM.backproject(*CAM.project(X, Z))
        assert z == pytest.approx([X, Z])
        assert math.sqrt(R[1, 1]) == pytest.approx(CAM.sigma_Z(Z))


def test_world_camera_frames_are_inverse():
    ego = EgoState(5.0, -2.0, 8.0, 1.1)
    zw, _ = camera_to_world(np.array([-1.5, 12.0]), np.eye(2), ego)
    assert world_to_camera(zw, ego) == pytest.approx([-1.5, 12.0])


# ---- UKF -------------------------------------------------------------------

def test_ukf_tracks_constant_turn():
    rng = np.random.default_rng(2)
    truth = np.array([0.0, 10.0, 5.0, math.pi / 2, -0.4])
    R = np.diag([0.05**2, 0.2**2])
    z0 = truth[:2] + rng.multivariate_normal([0, 0], R)
    trk = CTRVUKF.from_detection(ActorClass.AUTO_RICKSHAW, z0, R, heading=math.pi / 2, speed=4.0)
    dt = 1 / 15
    for _ in range(60):
        truth = ctrv_step(truth, dt)
        trk.predict(dt)
        trk.update(truth[:2] + rng.multivariate_normal([0, 0], R), R)
    assert np.linalg.norm(trk.position - truth[:2]) < 0.3
    assert trk.x[2] == pytest.approx(truth[2], abs=0.4)
    assert trk.x[4] == pytest.approx(truth[4], abs=0.1)
    assert np.all(np.linalg.eigvalsh(trk.P) > 0)


def test_class_agility_widens_prediction_cone():
    x0 = np.array([0.0, 10.0, 6.0, math.pi / 2, 0.0])
    P0 = np.diag([0.1, 0.1, 0.5, 1e-4, 1e-4])  # converged heading, so noise dominates

    def lateral_spread(cls):
        trk = CTRVUKF(cls, x0, P0)
        pts, _ = trk.augmented_sigma_points(lam=1.0, noise_scale=trk.step_dt / 2.5)
        return np.ptp(rollout(pts, trk.agility, 0.1, 25)[-1, :, 0])

    assert lateral_spread(ActorClass.AUTO_RICKSHAW) > 3 * lateral_spread(ActorClass.BUS)


# ---- P_conf ----------------------------------------------------------------

def _track(x, cls=ActorClass.AUTO_RICKSHAW):
    return CTRVUKF(cls, np.array(x, float), np.diag([0.05, 0.05, 0.2, 1e-3, 1e-3]))


def test_pconf_head_on_is_certain():
    ego = EgoState(0, 0, 8.0, math.pi / 2)
    trk = _track([0.0, 15.0, 5.0, -math.pi / 2, 0.0])
    assert conflict_probability(trk, ego) == pytest.approx(1.0)
    assert track_ttc_2d(trk, ego) == pytest.approx((15 - 2.5) / 13)


def test_pconf_far_parallel_car_is_zero():
    ego = EgoState(0, 0, 8.0, math.pi / 2)
    trk = _track([8.0, 5.0, 8.0, math.pi / 2, 0.0], ActorClass.CAR)
    assert conflict_probability(trk, ego) == 0.0
    assert track_ttc_2d(trk, ego) == math.inf


def test_pconf_monte_carlo_agrees_on_clear_cases():
    ego = EgoState(0, 0, 8.0, math.pi / 2)
    rng = np.random.default_rng(3)
    assert conflict_probability(_track([0.0, 15.0, 5.0, -math.pi / 2, 0.0]), ego,
                                n_samples=2000, rng=rng) > 0.95
    assert conflict_probability(_track([8.0, 5.0, 8.0, math.pi / 2, 0.0]), ego,
                                n_samples=2000, rng=rng) < 0.05


def test_pconf_rejects_negative_weights():
    with pytest.raises(ValueError):
        conflict_probability(_track([0, 15, 5, 0, 0]), EgoState(0, 0, 8, 0), lam=-4.0)


# ---- Agility estimation ----------------------------------------------------

def test_estimate_agility_recovers_known_spread():
    rng = np.random.default_rng(4)
    sigma_a, sigma_psidd, dt = 0.8, 0.3, 0.1
    tracks = []
    for _ in range(60):
        x = np.array([0.0, 0.0, 8.0, rng.uniform(-math.pi, math.pi), 0.0])
        pts = [x[:2]]
        for _ in range(10):  # ten 1-second blocks of constant acceleration inputs
            nu = rng.normal([0, 0], [sigma_a, sigma_psidd])
            for _ in range(10):
                x = ctrv_step(x, dt, nu)
                pts.append(x[:2])
        tracks.append(np.array(pts))
    est_a, est_psidd = estimate_agility(tracks, dt)
    assert est_a == pytest.approx(sigma_a, rel=0.15)
    assert est_psidd == pytest.approx(sigma_psidd, rel=0.15)


def test_every_class_has_agility():
    assert set(CLASS_AGILITY) == set(ActorClass)


# =============================================================================
# Stage B: lane-free calibration from vehicle widths
# =============================================================================

def test_calibration_widths_from_brief():
    assert CALIBRATION_WIDTH[ActorClass.CAR] == 1.75
    assert CALIBRATION_WIDTH[ActorClass.AUTO_RICKSHAW] == 1.35


def test_bbox_width_obeys_linear_law():
    # w / W_c = (1/h) y_b - y_h/h for any object standing on the road
    for X, Z, W in [(-3.0, 8.0, 1.75), (2.0, 25.0, 1.35), (0.0, 40.0, 1.75)]:
        x1, _, x2, yb = DASHCAM.bbox(X, Z, W, 1.5)
        assert (x2 - x1) / W == pytest.approx(yb / DASHCAM.h - DASHCAM.y_h / DASHCAM.h)


def test_calibration_exact_on_clean_boxes():
    rng = np.random.default_rng(10)
    Z = rng.uniform(6, 45, 40)
    W = np.where(rng.random(40) < 0.5, 1.75, 1.35)
    boxes = [DASHCAM.bbox(0.0, z, w, 1.5) for z, w in zip(Z, W)]
    cal = calibrate_from_vehicles([b[3] for b in boxes], [b[2] - b[0] for b in boxes], W)
    assert cal.slope == pytest.approx(1 / DASHCAM.h, rel=1e-4)
    assert cal.intercept == pytest.approx(-DASHCAM.y_h / DASHCAM.h, rel=1e-4)
    assert cal.h == pytest.approx(DASHCAM.h, rel=1e-4)
    assert cal.y_h == pytest.approx(DASHCAM.y_h, abs=0.05)


# Tolerances ~3 sigma of each method's spread over 30 seeds (Huber h sd 2.3 cm,
# RANSAC h sd 4.1 cm; both unbiased), with 15% outliers in the traffic.
@pytest.mark.parametrize("method, h_tol, yh_tol", [("huber", 0.07, 2.5), ("ransac", 0.12, 4.0)])
def test_calibration_recovers_camera_from_noisy_traffic(method, h_tol, yh_tol):
    rng = np.random.default_rng(11)
    dets = simulate_calibration_traffic(DASHCAM, 300, rng, outlier_fraction=0.15)
    cal = calibrate_from_detections(dets, method=method)
    assert cal.h == pytest.approx(DASHCAM.h, abs=h_tol)
    assert cal.y_h == pytest.approx(DASHCAM.y_h, abs=yh_tol)
    assert 0.7 < cal.inlier_fraction <= 1.0


def test_robust_fit_beats_least_squares_with_mislabelled_trucks():
    """One-sided contamination (trucks labelled 'car') biases OLS, not Huber/RANSAC."""
    rng = np.random.default_rng(12)
    yb, w, W = [], [], []
    for i in range(200):
        Z = rng.uniform(6, 40)
        real = 2.5 if i % 4 == 0 else 1.75   # 25% are trucks the detector called cars
        box = DASHCAM.bbox(0.0, Z, real, 1.5) + rng.normal(0, 1.0, 4)
        yb.append(box[3])
        w.append(box[2] - box[0])
        W.append(1.75)
    slope, _ = np.polyfit(yb, np.array(w) / np.array(W), 1)
    ols_err = abs(1 / slope - DASHCAM.h)
    for method in ("huber", "ransac"):
        cal = calibrate_from_vehicles(yb, w, W, method=method)
        assert abs(cal.h - DASHCAM.h) < 0.5 * ols_err
        assert abs(cal.h - DASHCAM.h) < 0.08


def test_pixel_to_ground_matches_formulas():
    x, yb = 900.0, 480.0
    X, Z = DASHCAM.pixel_to_ground(x, yb)
    d = yb - DASHCAM.y_h
    assert Z == pytest.approx(DASHCAM.f * DASHCAM.h / d)
    assert X == pytest.approx((x - DASHCAM.cx) * DASHCAM.h / d)
    assert DASHCAM.sigma_Z(Z) == pytest.approx(Z**2 * DASHCAM.sigma_y / (DASHCAM.f * DASHCAM.h))
    # Range error grows with distance squared.
    assert DASHCAM.sigma_Z(30.0) / DASHCAM.sigma_Z(15.0) == pytest.approx(4.0)


def test_pixel_above_horizon_has_no_ground_point():
    with pytest.raises(ValueError):
        DASHCAM.pixel_to_ground(640.0, DASHCAM.y_h - 1.0)


def test_backproject_bbox_uses_bottom_centre():
    z, _ = DASHCAM.backproject_bbox(DASHCAM.bbox(-2.0, 18.0, 1.35, 1.75))
    assert z == pytest.approx([-2.0, 18.0])


# =============================================================================
# CTRV transition: exact formulas from the brief
# =============================================================================

def test_ctrv_turning_matches_closed_form():
    X, Y, v, psi, w, dt = 1.0, 2.0, 7.0, 0.4, 0.6, 0.25
    out = ctrv_step([X, Y, v, psi, w], dt)
    assert out[0] == pytest.approx(X + v / w * (math.sin(psi + w * dt) - math.sin(psi)))
    assert out[1] == pytest.approx(Y + v / w * (-math.cos(psi + w * dt) + math.cos(psi)))
    assert out[2:] == pytest.approx([v, psi + w * dt, w])


def test_ctrv_straight_branch_at_threshold():
    # |psi_dot| <= 1e-4 uses the straight-line update exactly.
    v, psi, dt = 10.0, 0.7, 0.5
    out = ctrv_step([0.0, 0.0, v, psi, 1e-4], dt)
    assert out[0] == pytest.approx(v * math.cos(psi) * dt, abs=1e-12)
    assert out[1] == pytest.approx(v * math.sin(psi) * dt, abs=1e-12)
    assert out[3] == pytest.approx(psi + 1e-4 * dt)
    # Just above the threshold the turning branch takes over and agrees closely.
    turn = ctrv_step([0.0, 0.0, v, psi, 1.5e-4], dt)
    assert turn[:2] == pytest.approx(out[:2], abs=1e-3)


# =============================================================================
# Stage C: filterpy UKF with class-specific agility
# =============================================================================

def test_ukf_is_filterpy_with_merwe_points():
    trk = _track([0.0, 10.0, 5.0, 1.0, 0.0])
    assert isinstance(trk.kf, UnscentedKalmanFilter)
    assert isinstance(trk.kf.points_fn, MerweScaledSigmaPoints)


def test_agility_priors_match_brief():
    for c in (ActorClass.AUTO_RICKSHAW, ActorClass.TWO_WHEELER):
        assert (CLASS_AGILITY[c].sigma_a, CLASS_AGILITY[c].sigma_psidd) == (2.6, 1.2)
    for c in (ActorClass.BUS, ActorClass.TRUCK):
        assert (CLASS_AGILITY[c].sigma_a, CLASS_AGILITY[c].sigma_psidd) == (1.2, 0.25)
    assert ActorClass.parse("auto-rickshaw") is ActorClass.AUTO_RICKSHAW
    assert ActorClass.parse("rider") is ActorClass.TWO_WHEELER


def test_process_noise_is_class_specific():
    dt, psi = 0.1, 0.3
    Qa = process_noise(CLASS_AGILITY[ActorClass.AUTO_RICKSHAW], psi, dt)
    Qt = process_noise(CLASS_AGILITY[ActorClass.TRUCK], psi, dt)
    assert Qa[2, 2] / Qt[2, 2] == pytest.approx((2.6 / 1.2) ** 2)    # speed
    assert Qa[4, 4] / Qt[4, 4] == pytest.approx((1.2 / 0.25) ** 2)   # yaw rate
    assert np.all(np.linalg.eigvalsh(Qa) >= -1e-15)
    # The filter really uses it: after a predict its Q is the class Q at its heading.
    trk = CTRVUKF(ActorClass.AUTO_RICKSHAW, np.array([0, 0, 5.0, psi, 0.0]), np.eye(5) * 0.1)
    trk.predict(dt)
    assert trk.Q == pytest.approx(process_noise(trk.agility, psi, dt))


def test_process_noise_follows_heading():
    ag = CLASS_AGILITY[ActorClass.AUTO_RICKSHAW]
    Q_east, Q_north = process_noise(ag, 0.0, 0.1), process_noise(ag, math.pi / 2, 0.1)
    assert Q_east[0, 0] > 0 and Q_east[1, 1] == pytest.approx(0.0, abs=1e-15)
    assert Q_north[1, 1] > 0 and Q_north[0, 0] == pytest.approx(0.0, abs=1e-15)


def _track_swerve(cls, seed):
    """Auto weaving +-0.6 rad/s (2.5 s period) 12 m ahead, seen through the dashcam."""
    rng = np.random.default_rng(seed)
    ego = EgoState(0.0, 0.0, 0.0, math.pi / 2)
    dt = 1 / 15
    x = np.array([-1.0, 12.0, 6.0, math.pi / 2, 0.0])
    trk, pos_err, hdg_err, w_est, w_true = None, [], [], [], []
    for k in range(75):
        x[4] = 0.6 * math.sin(2 * math.pi * k * dt / 2.5)
        xp, yb = DASHCAM.project(*world_to_camera(x[:2], ego))
        zc, Rc = DASHCAM.backproject(xp + rng.normal(0, 1.5), yb + rng.normal(0, 1.5))
        z, R = camera_to_world(zc, Rc, ego)
        if trk is None:
            trk = CTRVUKF.from_detection(cls, z, R, heading=math.pi / 2, speed=5.0)
        else:
            trk.predict(dt)
            trk.update(z, R)
            if k > 15:  # after convergence
                pos_err.append(np.linalg.norm(trk.position - x[:2]))
                hdg_err.append(float(wrap_angle(trk.x[3] - x[3])))
                w_est.append(trk.x[4])
                w_true.append(x[4])
        x = ctrv_step(x, dt)

    def rms(a):
        return float(np.sqrt(np.mean(np.square(a))))

    return trk, rms(pos_err), rms(hdg_err), float(np.corrcoef(w_est, w_true)[0, 1]), x


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ukf_tracks_swerving_auto(seed):
    trk, pos_rms, hdg_rms, yaw_corr, truth = _track_swerve(ActorClass.AUTO_RICKSHAW, seed)
    assert pos_rms < 0.6
    assert hdg_rms < 0.2
    assert yaw_corr > 0.2                       # yaw-rate estimate follows the weave in phase
    assert trk.x[2] == pytest.approx(truth[2], abs=1.5)
    assert np.all(np.linalg.eigvalsh(trk.P) > 0)


def test_auto_agility_tracks_swerve_better_than_truck_agility():
    for seed in range(3):
        _, auto_pos, auto_hdg, _, _ = _track_swerve(ActorClass.AUTO_RICKSHAW, seed)
        _, trk_pos, trk_hdg, _, _ = _track_swerve(ActorClass.TRUCK, seed)
        assert auto_pos < 0.7 * trk_pos
        assert auto_hdg < 0.7 * trk_hdg


def test_ukf_oncoming_traffic_keeps_positive_speed():
    """Born with the ego heading as prior, an oncoming truck must flip, not track v < 0."""
    rng = np.random.default_rng(5)
    R = np.diag([0.1**2, 0.5**2])
    truth = np.array([6.0, 40.0, 11.0, -math.pi / 2, 0.0])
    trk = CTRVUKF.from_detection(ActorClass.TRUCK, truth[:2], R, heading=math.pi / 2, speed=13.0)
    for _ in range(20):
        truth = ctrv_step(truth, 1 / 15)
        trk.predict(1 / 15)
        trk.update(truth[:2] + rng.multivariate_normal([0, 0], R), R)
        assert trk.x[2] >= 0.0
    assert float(wrap_angle(trk.x[3] + math.pi / 2)) == pytest.approx(0.0, abs=0.2)


# =============================================================================
# Stage D/E: 2D TTC, crawl gate, baseline, alert policy
# =============================================================================

def test_ttc_formula_by_hand():
    r, v, R = np.array([3.0, 12.0]), np.array([-1.0, -6.0]), 2.5
    rv, vv, rr = r @ v, v @ v, r @ r
    expected = (-rv - math.sqrt(rv**2 - vv * (rr - R**2))) / vv
    assert ttc_2d(r, v, R) == pytest.approx(expected)


@pytest.mark.parametrize("v_rel", [[0.0, 0.4], [0.0, 0.0], [0.0, -3.0]])
def test_ttc_infinite_for_parallel_two_wheeler(v_rel):
    """Rider 3.2 m to the right on a parallel line: no conflict whatever the speed difference."""
    R = CLASS_AGILITY[ActorClass.TWO_WHEELER].radius + 1.4
    assert ttc_2d([3.2, 6.0], v_rel, R) == math.inf


def test_track_ttc_infinite_for_parallel_two_wheeler():
    ego = EgoState(0.0, 0.0, 13.3, math.pi / 2)
    rider = _track([3.2, 6.0, 13.7, math.pi / 2, 0.0], ActorClass.TWO_WHEELER)
    assert track_ttc_2d(rider, ego) == math.inf
    assert conflict_probability(rider, ego) < 0.5


@pytest.mark.parametrize("v_kmh, v_lat, suppress", [
    (17.0, 1.0, True), (17.0, -1.0, True),   # crawling, bunching: suppress
    (19.0, 1.0, False),                       # ego faster than 18 km/h
    (17.0, 1.3, False), (17.0, -1.3, False),  # actor moving sideways into the path
    (18.0, 0.0, False),                       # strict inequality at the boundary
])
def test_crawl_gate(v_kmh, v_lat, suppress):
    assert crawl_gate(v_kmh / 3.6, v_lat) is suppress


def test_crawl_gate_suppresses_bunching_but_not_squeeze_in():
    ego = EgoState(0.0, 0.0, 3.0, math.pi / 2)          # 10.8 km/h
    car = _track([0.0, 5.0, 1.0, math.pi / 2, 0.0], ActorClass.CAR)
    gated, ungated = assess(car, ego), assess(car, ego, use_crawl_gate=False)
    assert gated.crawl and not gated.bharat
    assert ungated.bharat and ungated.baseline
    # An auto cutting in sideways at ~2 m/s is not crawl-gated.
    auto = _track([-3.5, 3.5, 2.5, math.radians(10), 0.0])
    a = assess(auto, ego)
    assert not a.crawl and a.bharat


def test_ttc_1d_only_ahead_and_closing():
    assert ttc_1d(10.0, -4.0, 2.0) == pytest.approx(2.0)
    assert ttc_1d(10.0, 1.0, 2.0) == math.inf
    assert ttc_1d(-3.0, -4.0, 2.0) == math.inf


def test_alert_policy_two_of_three_and_cooldown():
    pol = AlertPolicy(k=2, n=3, cooldown=4.0)
    outs = [pol.update(i * 0.1, "a", raw, "Auto, left!", "left")
            for i, raw in enumerate([True, False, True])]
    assert [o[0] for o in outs] == [False, False, True]
    assert outs[2][1].message == "Auto, left!"
    pol.update(0.3, "a", False)
    pol.update(0.4, "a", False)                                      # clears
    pol.update(0.5, "a", True)
    active, ev = pol.update(0.6, "a", True)
    assert active and ev is None                                     # within cooldown: silent
    assert pol.update(0.6, "b", True) == (False, None)              # per-actor history


# =============================================================================
# End-to-end synthetic scenarios (bounding boxes -> calibration -> UKF -> alerts)
# =============================================================================

def test_hero_cut_in_warns_more_than_a_second_before_the_corridor():
    res = run_scenario(collision_course_cut_in(30.0), record=False)
    auto = res.summaries["auto-1"]
    assert auto.first_bharat is not None
    assert auto.corridor_entry - auto.first_bharat > 1.0      # conservative bound on lead
    if auto.first_baseline is not None:
        assert auto.first_baseline - auto.first_bharat > 1.0
    for quiet in ("bike-1", "truck-1"):                       # parallel rider, oncoming truck
        s = res.summaries[quiet]
        assert s.first_bharat is None and s.first_baseline is None
    assert res.summaries["bike-1"].min_ttc_2d == math.inf
    assert [e["message"] for e in res.events if e["system"] == "bharat"] == ["Auto, left!"]


def test_scenario_calibration_close_to_truth():
    cal = run_scenario(collision_course_cut_in(30.0), record=False).calibration
    assert cal["h_fit"] == pytest.approx(cal["h_true"], abs=0.06)
    assert cal["y_h_fit"] == pytest.approx(cal["y_h_true"], abs=3.0)


def test_crawl_gate_cuts_jam_nuisance_alerts():
    gated = run_scenario(crawl_jam(True), record=False).summaries
    ungated = run_scenario(crawl_jam(False), record=False).summaries
    assert gated["car-1"].baseline_cues >= 2          # the 1D baseline nags in the jam
    assert gated["car-1"].bharat_cues == 0            # gate on: bunching is not a hazard
    assert ungated["car-1"].bharat_cues >= 2          # gate off: the 2D rule nags as well
    assert gated["auto-2"].bharat_cues == 1           # the sideways squeeze-in still gets a cue


def test_merge_then_straighten_is_a_known_limitation():
    """Documented counter-example: a swerve that straightens out gives the 2D rule no edge."""
    s = run_scenario(merge_then_straighten(), record=False).summaries["auto-1"]
    assert s.first_baseline is not None and s.first_bharat is not None
    assert s.lead_gain < 0


def test_simulation_is_deterministic():
    a = run_scenario(collision_course_cut_in(25.0), record=False)
    b = run_scenario(collision_course_cut_in(25.0), record=False)
    assert a.events == b.events
    assert vars(a.summaries["auto-1"]) == vars(b.summaries["auto-1"])
