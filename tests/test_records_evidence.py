import json
import math

import numpy as np
import pytest
from filterpy.kalman import UnscentedKalmanFilter

from bharat_ttc import CTRVUKF, ActorClass, EgoState, ctrv_step
from bharat_ttc.evidence import benchmark, crawl_comparison, record_sample, sweep
from bharat_ttc.records import NearMissRecord, drac, h3_cell, local_to_latlon
from bharat_ttc.sim import collision_course_cut_in, run_scenario
from bharat_ttc.ukf import N_X, residual_x, state_mean


# ---- near-miss records (Layer 2 input) -------------------------------------------

def _record(**kw):
    base = dict(v=1, h3=h3_cell(28.46, 77.03), t=1793334601, ego_kmh=48.0, min_ttc_s=1.54,
                ego_accel=-0.4, drac=5.25, p_conf=0.94, actor="auto_rickshaw", side="left", light="day")
    base.update(kw)
    return NearMissRecord(**base)


def test_record_is_within_the_deck_size_budget():
    r = _record()
    assert r.sizes()["json_bytes"] <= 300          # deck: "~300 bytes"
    assert r.sizes()["binary_bytes"] == 32
    assert "video" not in r.to_json()


def test_record_binary_round_trip():
    r = _record(trigger="hard_brake")
    back = NearMissRecord.from_bytes(r.to_bytes())
    assert back.h3 == r.h3 and back.actor == r.actor and back.side == r.side and back.trigger == "hard_brake"
    assert back.ego_kmh == pytest.approx(r.ego_kmh, abs=0.05)
    assert back.min_ttc_s == pytest.approx(r.min_ttc_s, abs=1e-3)
    assert back.drac == pytest.approx(r.drac, abs=0.01)
    assert back.p_conf == pytest.approx(r.p_conf, abs=1 / 255)


def test_h3_cell_is_resolution_9():
    import h3
    assert h3.get_resolution(h3_cell(28.46, 77.03)) == 9


def test_local_offset_and_drac():
    lat, lon = local_to_latlon(0.0, 111.32, 28.0, 77.0)
    assert lat == pytest.approx(28.001, abs=1e-6) and lon == pytest.approx(77.0)
    assert drac(10.0, 2.0) == pytest.approx(5.0)
    assert drac(10.0, math.inf) == 0.0


def test_every_bharat_alert_emits_a_record():
    res = run_scenario(collision_course_cut_in(30.0), record=False)
    cues = [e for e in res.events if e["system"] == "bharat"]
    assert cues and all("record" in e for e in cues)
    rec = json.loads(cues[0]["record"])
    assert rec["actor"] == "auto_rickshaw" and rec["side"] == "left"
    assert rec["ego_kmh"] == pytest.approx(48.0, abs=0.1)
    assert rec["min_ttc_s"] <= 1.8 and rec["drac"] > 0


# ---- speed-ups keep the maths identical ----------------------------------------------

def test_vectorised_ukf_matches_stock_filterpy():
    x0 = np.array([1.0, 12.0, 6.0, 1.2, -0.3])
    P0 = np.diag([0.2, 0.3, 1.0, 0.1, 0.05])
    ours = CTRVUKF(ActorClass.AUTO_RICKSHAW, x0, P0)
    stock = UnscentedKalmanFilter(N_X, 2, 0.1, hx=lambda x: x[:2], fx=lambda x, dt: ctrv_step(x, dt),
                                  points=ours.kf.points_fn, x_mean_fn=state_mean, residual_x=residual_x)
    stock.x, stock.P = x0.copy(), P0.copy()
    ours.predict(1 / 15)
    stock.Q = ours.Q
    stock.predict(dt=1 / 15)
    assert ours.x == pytest.approx(stock.x, abs=1e-12)
    assert ours.P == pytest.approx(stock.P, abs=1e-12)


@pytest.mark.parametrize("w", [0.0, 0.35, -0.8])
def test_closed_form_ego_rollout_matches_stepping(w):
    ego = EgoState(2.0, -1.0, 12.0, 0.7, w)
    s, steps = ego.state, [ego.state[:2]]
    for _ in range(25):
        s = ctrv_step(s, 0.1)
        steps.append(s[:2])
    assert ego.rollout(0.1, 25) == pytest.approx(np.array(steps), abs=1e-9)


# ---- evidence ------------------------------------------------------------------------

def test_sweep_structure_and_monotone_trend():
    sw = sweep(angles=(10, 30), seeds=(1, 2))
    assert sw["runs"] == 4 and len(sw["rows"]) == 2
    lo, hi = sw["rows"]
    assert lo["lead_vs_entry_median"] < hi["lead_vs_entry_median"]   # shallow drifts gain less
    assert hi["lead_vs_entry_median"] > 1.0


def test_crawl_comparison_matches_gate_story():
    cr = crawl_comparison()
    assert cr["gate_on"]["bunching_bharat"] == 0 < cr["gate_on"]["bunching_baseline"]
    assert cr["gate_off"]["bunching_bharat"] > 0
    assert cr["gate_on"]["squeeze_bharat"] == 1


def test_benchmark_reports_sane_numbers():
    b = benchmark(n_frames=40, repeats=2)
    assert 0 < b["ms_best_repeat"] <= b["ms_worst_repeat"]
    assert b["ms_total"] <= b["ms_total_p95"]
    assert b["actors_in_budget"] >= 1


def test_record_sample_fits_daily_budget():
    r = record_sample()
    assert r["records_per_mb_json"] >= 3000


def test_version_numbers_agree():
    import pathlib

    import bharat_ttc

    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    from web.server import app

    root = pathlib.Path(__file__).resolve().parent.parent
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert bharat_ttc.__version__ == project["version"] == app.version
