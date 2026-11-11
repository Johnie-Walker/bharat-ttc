/* Bharat-TTC dashboard v0.3: tabs, Evidence and Fleet panels, guided judge tour.
   Uses globals from app.js: state, $, C, hexA, loadScenario, seek, togglePlay, pause, render, sizeLanes. */
"use strict";

const ev = { data: null, charts: {}, polling: false };
let currentTab = "live";

// ------------------------------------------------------------------ tabs --
function showTab(name) {
  currentTab = name;
  for (const b of document.querySelectorAll(".tab-btn")) {
    const on = b.dataset.tab === name;
    b.classList.toggle("active", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  }
  for (const p of document.querySelectorAll(".tab-panel")) p.hidden = p.id !== `tab-${name}`;
  if (name === "live") {
    sizeLanes();
    for (const ch of Object.values(state.charts)) ch.resize();
    render();
  } else {
    pause();
  }
  if (name === "evidence") renderEvidence();
  if (name === "fleet") renderFleet();
}
for (const b of document.querySelectorAll(".tab-btn")) b.addEventListener("click", () => showTab(b.dataset.tab));

// -------------------------------------------------------------- evidence --
async function loadEvidence() {
  if (ev.data || ev.polling) return ev.data;
  ev.polling = true;
  for (;;) {
    const r = await fetch("/api/evidence");
    if (r.status === 200) { ev.data = await r.json(); break; }
    if (r.status !== 202) { $("ev-loading").textContent = `Evidence failed (${r.status}).`; ev.polling = false; return null; }
    await new Promise((res) => setTimeout(res, 1500));
  }
  ev.polling = false;
  if (currentTab === "evidence") renderEvidence();
  if (currentTab === "fleet") renderFleet();
  return ev.data;
}

const f2 = (v, u = " s") => (v === null || v === undefined ? "n/a" : `${v.toFixed(2)}${u}`);
const sgn = (v) => (v > 0 ? "+" : "") + v.toFixed(2);
const tile = (v, k, cls = "") => `<div class="tile ${cls}"><span class="tile-v">${v}</span><span class="tile-k">${k}</span></div>`;

function renderEvidence() {
  if (!ev.data) { loadEvidence(); return; }
  const d = ev.data, sw = d.sweep, cr = d.crawl, ab = d.ablation, bm = d.benchmark, rc = d.record;
  $("ev-loading").hidden = true;

  // scorecard
  const rows = [
    ["H1", "Earlier warning on angular cut-ins", "Median gain ≥ 1.0 s",
     `<b>${sgn(sw.median_lead_vs_entry)} s</b> median over ${sw.runs} runs vs the earliest any corridor rule could fire ` +
     `(${Math.round(sw.share_over_1s * 100)}% of runs above 1 s); ${sgn(sw.median_lead_vs_baseline)} s vs the tracked baseline`,
     "met", "Met in simulation"],
    ["H2", "Fewer nuisance alerts in slow traffic, equal or better recall", "≥ 50% fewer alerts/hour below 20 km/h",
     `One ${cr.duration_s} s jam: bunching cues <b>${cr.gate_on.bunching_baseline} → ${cr.gate_on.bunching_bharat}</b>; ` +
     `sideways squeeze-in still caught (${cr.gate_on.squeeze_bharat} of 1)`, "partial", "Directional, 1 scenario"],
    ["H3", "Runs on hardware fleets already buy", "≥ 15 fps on Jetson Orin Nano class",
     `Layer 1 only: median <b>${bm.ms_total.toFixed(2)} ms</b> per actor-frame (p95 ${bm.ms_total_p95.toFixed(2)}) on a laptop CPU, ` +
     `~${bm.actors_in_budget_p95}+ actors per 15 fps frame at p95. Detector and Jetson not measured`, "partial", "Layer 1 only"],
    ["H4", "Near-miss map is stable, not noise", "Split-half rank correlation ≥ 0.5",
     "Needs fleet GPS logs (ThirdEye Delhi NCR); the map is not built in this prototype", "open", "Not yet measured"],
    ["H5", "Cheap at fleet scale", "< 1 MB upload per vehicle per day",
     `Record is <b>${rc.json_bytes} B</b> JSON / ${rc.binary_bytes} B packed: ~${rc.records_per_mb_json.toLocaleString("en-IN")} alerts/day fit in 1 MB`,
     "met", "Met for record size"],
  ];
  $("scorecard").innerHTML = `<thead><tr><th>#</th><th>Hypothesis</th><th>Deck target</th><th>What this prototype shows</th><th>Status</th></tr></thead><tbody>` +
    rows.map((r) => `<tr><td class="res">${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td><td><span class="status ${r[4]}">${r[5]}</span></td></tr>`).join("") +
    `</tbody>`;

  // H1 sweep chart
  if (!ev.charts.sweep) {
    const pts = (key) => sw.rows.map((r) => ({ x: r.angle, y: r[key] }));
    const neutral = C.ink2;
    ev.charts.sweep = new Chart($("sweep-chart"), {
      type: "line",
      data: { datasets: [
        { label: "seed range (min)", data: pts("lead_vs_entry_min"), borderColor: "transparent", pointRadius: 0, fill: false },
        { label: "spread across 5 seeds", data: pts("lead_vs_entry_max"), borderColor: "transparent", backgroundColor: hexA(neutral, 0.14), pointRadius: 0, fill: "-1" },
        { label: "Lead vs tracked 1D baseline (median)", data: pts("lead_vs_baseline_median"), borderColor: C.bharat, backgroundColor: C.bharat, borderWidth: 2, pointRadius: 5, pointHoverRadius: 7 },
        { label: "Lead vs corridor entry, lower bound (median)", data: pts("lead_vs_entry_median"), borderColor: neutral, backgroundColor: neutral, borderWidth: 2, pointRadius: 5, pointHoverRadius: 7 },
        { label: "Deck target H1: 1.0 s", data: [{ x: 8, y: 1 }, { x: 47, y: 1 }], borderColor: hexA(C.green, 0.8), borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: "nearest", axis: "x", intersect: false },
        scales: {
          x: { type: "linear", min: 8, max: 47, title: { display: true, text: "cut-in swing angle (degrees)", color: C.muted },
               ticks: { color: C.muted, stepSize: 5 }, grid: { color: "rgba(185,190,195,0.08)" } },
          y: { min: 0, max: 1.6, title: { display: true, text: "warning lead time (s)", color: C.muted },
               ticks: { color: C.muted }, grid: { color: "rgba(185,190,195,0.08)" } },
        },
        plugins: {
          legend: { labels: { color: C.ink2, filter: (it) => it.text !== "seed range (min)", boxWidth: 14, boxHeight: 3 } },
          tooltip: { backgroundColor: "rgba(17,19,21,0.95)", borderColor: C.line, borderWidth: 1,
                     filter: (it) => it.datasetIndex >= 2 && it.datasetIndex <= 3,
                     callbacks: { title: (it) => `${it[0].parsed.x}° swing`, label: (it) => `${it.dataset.label}: ${it.parsed.y === null ? "n/a" : "+" + it.parsed.y.toFixed(2) + " s"}` } },
        },
      },
    });
  }
  const weak = sw.rows.filter((r) => r.lead_vs_entry_median !== null && r.lead_vs_entry_median < 1).map((r) => `${r.angle}°`);
  $("sweep-note").textContent = `Lower bound: no corridor rule can fire before the auto is inside the corridor. ` +
    (weak.length ? `Below 1 s at ${weak.join(", ")}: shallow drifts enter the corridor early. ` : "") +
    `Conflict at t = ${sw.t_conflict} s if nobody brakes.`;

  $("jam-tiles").innerHTML =
    tile(cr.gate_on.bunching_baseline, `<span class="sw" style="background:${C.base}"></span>1D baseline beeps on normal bunching`, "bad") +
    tile(cr.gate_on.bunching_bharat, `<span class="sw" style="background:${C.bharat}"></span>Bharat-TTC cues, crawl gate on`, "good") +
    tile(cr.gate_off.bunching_bharat, `<span class="sw" style="background:${C.bharat}"></span>Bharat-TTC cues, crawl gate off (ablation)`) +
    tile(`${cr.gate_on.squeeze_bharat} <small>of 1</small>`, "sideways squeeze-in still warned (gate on)", "good");

  const sa = ab.swerve_tracking.auto_rickshaw, st = ab.swerve_tracking.truck;
  $("abl-tiles").innerHTML =
    tile(f2(ab.cut_in.auto_prior), "cut-in cue, auto agility prior") +
    tile(f2(ab.cut_in.truck_prior), "cut-in cue, truck agility prior") +
    tile(`${sa.pos_rms.toFixed(2)} <small>m</small>`, `swerve tracking error, auto prior (heading ${sa.heading_rms_deg.toFixed(0)}°)`, "good") +
    tile(`${st.pos_rms.toFixed(2)} <small>m</small>`, `same swerve, truck prior (heading ${st.heading_rms_deg.toFixed(0)}°)`, "bad");
  $("abl-note").textContent = `Agility does not move the cut-in warning here (the lead comes from 2D geometry). It makes swerve ` +
    `tracking ${(st.pos_rms / sa.pos_rms).toFixed(1)}× more accurate (RMS over ${ab.swerve_seeds} seeds). ` +
    `Counter-example: merge-then-straighten, the baseline is first by ${(-ab.merge.lead).toFixed(2)} s.`;

  $("bench-tiles").innerHTML =
    tile(`${bm.ms_total.toFixed(2)} <small>ms</small>`, "median per tracked actor per frame (UKF + TTC₂D + P_conf)") +
    tile(`${bm.ms_total_p95.toFixed(2)} <small>ms</small>`, "95th percentile") +
    tile(`~${bm.actors_in_budget_p95}`, "actors per 15 fps frame at p95 (67 ms budget)", "good") +
    tile(`${bm.calibration_fit_ms.toFixed(0)} <small>ms</small>`, "lane-free calibration refit, 240 boxes");
  $("bench-note").textContent = `Pure Python/NumPy on ${bm.cpu}, Python ${bm.python}, ${bm.repeats} repeats × ` +
    `${bm.frames / bm.repeats} frames; repeat medians ranged ${bm.ms_best_repeat.toFixed(2)}–${bm.ms_worst_repeat.toFixed(2)} ms ` +
    `as the OS moved the process between performance and efficiency cores` +
    `${bm.power_throttling_disabled ? " (Windows power throttling disabled for this process)" : ""}. ` +
    `Excludes the detector; not a Jetson measurement.`;

  $("limits").innerHTML = d.limitations.map((l) => `<li>${l}</li>`).join("");
}

// ----------------------------------------------------------------- fleet --
function renderFleet() {
  const d = state.data;
  const recs = d ? d.events.filter((e) => e.system === "bharat" && e.record) : [];
  $("rec-scn").textContent = d ? d.scenario.title : "";
  $("records").innerHTML = recs.length
    ? recs.map((e) => {
        const r = JSON.parse(e.record);
        return `<div class="rec"><div class="rec-head"><span>t = <b>${e.t.toFixed(2)} s</b></span><span>cue <b>“${e.message}”</b></span>` +
          `<span>H3 cell <b>${r.h3}</b></span><span>DRAC <b>${r.drac.toFixed(2)} m/s²</b></span>` +
          `<span><b>${e.json_bytes} B</b> JSON · <b>${e.binary_bytes} B</b> packed</span></div><code>${e.record}</code></div>`;
      }).join("")
    : `<div class="rec">No Bharat-TTC alert in this scenario, so nothing is uploaded.</div>`;
  updateBudget();
}

function updateBudget() {
  const n = +$("alerts-day").value;
  $("o-alerts").textContent = n.toLocaleString("en-IN");
  const d = state.data;
  const first = d && d.events.find((e) => e.record);
  const jb = first ? first.json_bytes : ev.data ? ev.data.record.json_bytes : 204;
  const bb = first ? first.binary_bytes : 32;
  const max = 1.25e6;
  const row = (label, bytes) => {
    const pct = Math.min(100, (bytes / max) * 100);
    return `<div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill ${bytes > 1e6 ? "over" : ""}" style="width:${pct}%"></div>` +
      `<div class="limit" style="left:${(1e6 / max) * 100}%" title="1 MB target"></div></div><span>${(bytes / 1e6).toFixed(3)} MB</span></div>`;
  };
  $("budget-bars").innerHTML = row(`JSON (${jb} B)`, n * jb) + row(`packed (${bb} B)`, n * bb) +
    `<div class="bar-row"><span></span><span class="note" style="margin:0">white line = 1 MB per day target</span><span></span></div>`;
}
$("alerts-day").addEventListener("input", updateBudget);
window.afterLoad = () => { if (currentTab === "fleet") renderFleet(); };

// ------------------------------------------------------------ judge tour --
const tour = { i: -1, spot: null };

function hero() { return state.data.summaries[state.data.scenario.focus]; }
function actorAt(t, id) {
  const f = frameAt(t);
  return f.actors.find((a) => a.id === id);
}

const STEPS = [
  { tab: "live", scenario: "hero", seek: 0, spot: "dashcam",
    text: () => {
      const c = state.data.calibration;
      return `A truck at ${state.data.scenario.ego_speed_kmh} km/h. An auto ahead-left is about to swing across its path. ` +
        `There are no lane markings: camera height and horizon were fitted from the traffic itself ` +
        `(h = ${c.h_fit.toFixed(3)} m vs ${c.h_true.toFixed(3)} m true; horizon ${c.y_h_fit.toFixed(1)} vs ${c.y_h_true.toFixed(1)} px). Press Next to play.`;
    } },
  { tab: "live", scenario: "hero", playTo: () => hero().first_bharat + 0.02, spot: "dashcam",
    text: () => {
      const a = actorAt(hero().first_bharat, state.data.scenario.focus);
      const lat = a && a.track ? Math.abs(a.track.pos[0]).toFixed(1) : "?";
      return `Bharat-TTC speaks: “Auto, left!” at ${hero().first_bharat.toFixed(2)} s. The auto is still ${lat} m to the side, ` +
        `outside the blue corridor a lane-based system watches. On the radar its predicted cone already reaches the truck.`;
    } },
  { tab: "live", scenario: "hero", playTo: () => hero().first_baseline + 0.02, spot: "h2h",
    text: () => {
      const s = hero(), sc = state.data.scenario;
      return `Only now, at ${s.first_baseline.toFixed(2)} s, does the lane-based baseline beep: ${(sc.t_conflict - s.first_baseline).toFixed(2)} s before impact. ` +
        `Same detections, same tracker. Bharat-TTC was ${s.lead_gain.toFixed(2)} s earlier, about ${Math.round(s.lead_gain * sc.ego_speed_kmh / 3.6)} m more room to brake.`;
    } },
  { tab: "live", scenario: "jam", rate: 1, playTo: () => state.data.events.find((e) => e.system === "baseline").t + 0.02, spot: "h2h",
    text: () => `Stop-and-go jam under 15 km/h. The baseline beeps at a normal squeeze behind the car ahead. ` +
      `Bharat-TTC stays quiet: its crawl gate (ego below 18 km/h, sideways speed below 1.2 m/s) knows bunching is not a hazard.` },
  { tab: "live", scenario: "jam", rate: 1, playTo: () => state.data.events.find((e) => e.system === "bharat").t + 0.02, spot: "radar",
    text: () => {
      const beeps = state.data.events.filter((e) => e.system === "baseline").length;
      return `But an auto squeezing in sideways is a real hazard, and the gate lets it through: “Auto, left!”. ` +
        `Over the whole ${state.data.scenario.duration} s jam: ${beeps} nuisance beeps from the baseline, one useful cue from Bharat-TTC.`;
    } },
  { tab: "evidence", spot: "sweep-chart",
    text: () => ev.data
      ? `Not one cherry-picked clip: ${ev.data.sweep.runs} cut-ins over 8 angles and 5 noise seeds. Median lead ${sgn(ev.data.sweep.median_lead_vs_entry)} s ` +
        `over the earliest any lane rule could fire. Shallow drifts fall under 1 s, and that is shown, not hidden.`
      : "Evidence is still being computed on the server; it appears here in a few seconds." },
  { tab: "evidence", spot: "scorecard",
    text: () => `The deck’s five hypotheses, scored honestly: met in simulation, directional, or still waiting for METEOR clips, ` +
      `fleet GPS logs and a Jetson. The limits card lists where the method loses.` },
  { tab: "fleet", spot: "records",
    text: () => {
      const r = ev.data ? ev.data.record : { json_bytes: 204, binary_bytes: 32, records_per_mb_json: 4901 };
      return `Every alert leaves the truck as a ${r.json_bytes}-byte record (${r.binary_bytes} bytes packed): no video, no driver data. ` +
        `About ${r.records_per_mb_json.toLocaleString("en-IN")} fit in the 1 MB/day budget. Pooled per H3 cell, they become a near-miss map.`;
    } },
  { tab: "live", spot: "custom-form",
    text: () => "Your turn: set any cut-in angle, speeds or agility prior, and the server re-runs the full pipeline from bounding boxes." },
];

function clearSpot() { if (tour.spot) tour.spot.classList.remove("spot"); tour.spot = null; }
function spotlight(id) {
  clearSpot();
  let el = $(id);
  if (!el) return;
  el = el.closest(".card") || el;
  el.classList.add("spot");
  tour.spot = el;
  el.scrollIntoView({ block: "center", behavior: "smooth" });
}

async function enterStep(i, replay) {
  const s = STEPS[i];
  tour.i = i;
  state.stopAt = null; state.onStop = null;
  pause();
  if (currentTab !== s.tab) showTab(s.tab);
  if (s.scenario && (!state.data || state.data.scenario.key !== s.scenario)) {
    $("scenario").value = s.scenario;
    await loadScenario(s.scenario);
  }
  if (s.tab === "evidence" && !ev.data) await loadEvidence();
  $("tour-step").textContent = `Step ${i + 1} of ${STEPS.length}`;
  $("tour-back").disabled = i === 0;
  $("tour-next").textContent = i === STEPS.length - 1 ? "Finish ✓" : "Next ▸";
  spotlight(s.spot);
  if (s.seek !== undefined) seek(s.seek);
  if (s.playTo) {
    const target = Math.min(s.playTo(), state.data.scenario.duration);
    if (replay && state.t < target - 1e-6) {
      $("tour-text").textContent = "Watch the dashcam and radar…";
      $("tour-next").disabled = true;
      if (s.rate) { state.rateBackup = state.rate; state.rate = s.rate; }
      state.stopAt = target;
      state.onStop = () => {
        if (state.rateBackup) { state.rate = state.rateBackup; state.rateBackup = null; }
        $("tour-next").disabled = false;
        $("tour-text").textContent = s.text();
      };
      if (!state.playing) togglePlay();
      return;
    }
    seek(target);
  }
  $("tour-next").disabled = false;
  $("tour-text").textContent = s.text();
}

function startTour() {
  $("tour").hidden = false;
  ensureAudio();
  enterStep(0, true);
}
function endTour() {
  $("tour").hidden = true;
  clearSpot();
  state.stopAt = null; state.onStop = null;
  if (state.rateBackup) { state.rate = state.rateBackup; state.rateBackup = null; }
  pause();
}
$("tour-start").addEventListener("click", startTour);
$("tour-exit").addEventListener("click", endTour);
$("tour-next").addEventListener("click", () => (tour.i >= STEPS.length - 1 ? endTour() : enterStep(tour.i + 1, true)));
$("tour-back").addEventListener("click", () => { if (tour.i > 0) enterStep(tour.i - 1, false); });
window.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("tour").hidden) endTour(); });

// deep links: ?tab=evidence, ?tour=1
window.appReady.then(() => {
  const q = new URLSearchParams(location.search);
  if (["evidence", "fleet"].includes(q.get("tab"))) showTab(q.get("tab"));
  if (q.get("tour")) startTour();
  loadEvidence();
});
