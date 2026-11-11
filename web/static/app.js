/* Bharat-TTC judge dashboard. Plays back a simulator run frame by frame. */
"use strict";

const C = {
  asphalt: "#191C1F", panel: "#212529", panel2: "#2A2E33", line: "#3A4046",
  ink: "#EEF0F1", ink2: "#B9BEC3", muted: "#8A9099",
  yellow: "#F0B92B", red: "#F06A5F", green: "#6CC48E",
  bharat: "#BC8B14", base: "#4F8AE0",
};
const SIZE = {            // (width, height) m, as in bharat_ttc.camera.VEHICLE_SIZE
  person: [0.5, 1.65], two_wheeler: [0.75, 1.55], bicycle: [0.6, 1.6], auto_rickshaw: [1.35, 1.75],
  car: [1.75, 1.5], bus: [2.6, 3.2], truck: [2.5, 3.3], animal: [0.7, 1.4],
};
const RADIUS = { person: 0.5, two_wheeler: 0.8, bicycle: 0.8, auto_rickshaw: 1.1, car: 1.5, bus: 2.0, truck: 2.0, animal: 1.0 };
const LABEL = { person: "Pedestrian", two_wheeler: "Rider", bicycle: "Cycle", auto_rickshaw: "Auto", car: "Car", bus: "Bus", truck: "Truck", animal: "Animal" };
const EGO_RADIUS = 1.4;
const BLIND_ZONE = 3.8;

const $ = (id) => document.getElementById(id);
const state = { data: null, t: 0, playing: false, rate: 0.5, lastTs: null, charts: {}, chartActor: null, bannerTimer: null };

// ---------------------------------------------------------------- loading --
async function init() {
  const list = await (await fetch("/api/scenarios")).json();
  const sel = $("scenario");
  for (const s of list) sel.add(new Option(s.title, s.key));
  sel.addEventListener("change", () => { if (sel.value !== "custom") loadScenario(sel.value); });
  $("play").addEventListener("click", togglePlay);
  $("restart").addEventListener("click", () => { seek(0); if (!state.playing) togglePlay(); });
  $("rate").addEventListener("change", (e) => { state.rate = +e.target.value; });
  $("truth").addEventListener("change", render);
  $("scrub").addEventListener("input", (e) => {
    pause();
    seek((+e.target.value / 1000) * state.data.scenario.duration);
  });
  $("chart-actor").addEventListener("change", (e) => { state.chartActor = e.target.value; buildCharts(); render(); });
  setupCustomForm();
  window.addEventListener("resize", () => { sizeLanes(); render(); });
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !["INPUT", "SELECT", "BUTTON"].includes(document.activeElement.tagName)) {
      e.preventDefault(); togglePlay();
    }
  });
  // Deep links: /?scenario=jam&t=2.5
  const q = new URLSearchParams(location.search);
  const key = list.some((s) => s.key === q.get("scenario")) ? q.get("scenario") : list[0].key;
  sel.value = key;
  await loadScenario(key);
  if (q.has("t")) seek(+q.get("t") || 0);
}

async function loadScenario(key, payload) {
  pause();
  $("sc-title").textContent = "Simulating…";
  const res = payload
    ? await fetch("/api/custom", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
    : await fetch(`/api/simulate/${key}`);
  if (!res.ok) { $("sc-title").textContent = `Error ${res.status}`; return; }
  state.data = await res.json();
  const sc = state.data.scenario;
  $("sc-title").textContent = sc.title;
  $("sc-desc").textContent = sc.description;
  const ids = [...new Set(state.data.frames.flatMap((f) => f.actors.map((a) => a.id)))];
  const ca = $("chart-actor");
  ca.innerHTML = "";
  for (const id of ids) ca.add(new Option(id, id));
  state.chartActor = sc.focus && ids.includes(sc.focus) ? sc.focus : ids[0];
  ca.value = state.chartActor;
  // Radar extent: fit the farthest tracked actor, between a 14 m and a 42 m view.
  const far = Math.max(0, ...state.data.frames.flatMap((f) => f.actors.filter((a) => a.track).map((a) => a.track.pos[1])));
  const fwdMax = Math.min(42, Math.max(14, Math.ceil(Math.min(far, 40) + 4)));
  state.radar = { fwdMax, fwdMin: -fwdMax * 0.3 };
  renderKpis();
  renderEvents();
  buildCharts();
  buildCalibration();
  sizeLanes();
  seek(0);
  if (window.afterLoad) window.afterLoad();
}

// --------------------------------------------------------------- playback --
function togglePlay() {
  if (state.playing) return pause();
  if (state.t >= state.data.scenario.duration - 1e-6) seek(0);
  state.playing = true;
  state.lastTs = null;
  $("play").textContent = "❚❚ Pause";
  // Speech must start from a user gesture in most browsers; prime it here.
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  ensureAudio();
  requestAnimationFrame(tick);
}
function pause() {
  state.playing = false;
  $("play").textContent = "▶ Play";
}
function tick(ts) {
  if (!state.playing) return;
  if (state.lastTs !== null) {
    const prev = state.t;
    let next = Math.min(state.data.scenario.duration, prev + ((ts - state.lastTs) / 1000) * state.rate);
    const stop = state.stopAt;
    if (stop !== null && stop !== undefined && next >= stop) next = stop;  // guided tour checkpoint
    fireEvents(prev, next);
    state.t = next;
    if (stop !== null && stop !== undefined && next >= stop) {
      state.stopAt = null;
      pause();
      if (state.onStop) { const cb = state.onStop; state.onStop = null; cb(); }
    } else if (next >= state.data.scenario.duration) pause();
  }
  state.lastTs = ts;
  render();
  if (state.playing) requestAnimationFrame(tick);
}
function seek(t) {
  state.t = Math.max(0, Math.min(t, state.data.scenario.duration));
  render();
}
function frameAt(t) {
  const fr = state.data.frames;
  const i = Math.min(fr.length - 1, Math.max(0, Math.floor(t * state.data.scenario.fps + 1e-6)));
  return fr[i];
}

// ---------------------------------------------------- voice cues & beeps --
let audioCtx = null;
function ensureAudio() {
  try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); audioCtx.resume(); } catch (_) { audioCtx = null; }
}
function beep() {
  if (!audioCtx) return;
  const t0 = audioCtx.currentTime;
  for (let k = 0; k < 2; k++) {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.type = "square"; o.frequency.value = 1240;
    g.gain.setValueAtTime(0.0001, t0 + k * 0.16);
    g.gain.exponentialRampToValueAtTime(0.08, t0 + k * 0.16 + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + k * 0.16 + 0.11);
    o.connect(g).connect(audioCtx.destination);
    o.start(t0 + k * 0.16); o.stop(t0 + k * 0.16 + 0.12);
  }
}
function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  const voices = window.speechSynthesis.getVoices();
  const v = voices.find((x) => /en[-_]IN/i.test(x.lang)) || voices.find((x) => /^en/i.test(x.lang));
  if (v) u.voice = v;
  u.rate = 1.05; u.pitch = 1.0; u.volume = 1.0;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}
function fireEvents(prev, next) {
  for (const e of state.data.events) {
    if (e.t > prev && e.t <= next) {
      if (e.system === "bharat") {
        if ($("voice").checked) speak(e.message);
        const word = e.message.replace(/!$/, "").toUpperCase();
        showBanner(e.side === "left" ? `◀  ${word}` : e.side === "right" ? `${word}  ▶` : `▲  ${word}`, "bharat");
      } else {
        if ($("beep").checked) beep();
        showBanner("BASELINE: BRAKE!", "base");
      }
    }
  }
}
function showBanner(text, kind) {
  const b = $("banner");
  b.textContent = text;
  b.className = `banner show ${kind === "base" ? "base" : ""}`;
  clearTimeout(state.bannerTimer);
  state.bannerTimer = setTimeout(() => { b.className = "banner"; }, 1600);
}

// ----------------------------------------------------------------- render --
function render() {
  if (!state.data) return;
  const f = frameAt(state.t);
  $("clock-t").textContent = state.t.toFixed(2);
  $("clock-v").textContent = Math.round(f.ego.kmh);
  $("scrub").value = Math.round((state.t / state.data.scenario.duration) * 1000);
  drawDashcam(f);
  drawRadar(f);
  drawLanes();
  for (const ch of Object.values(state.charts)) ch.draw();
  for (const li of document.querySelectorAll("#events li[data-t]")) li.classList.toggle("past", +li.dataset.t <= state.t + 1e-6);
}

function actorState(a) {
  if (a.bharat) return "alert";
  if (a.risk && (a.risk.bharat_raw || a.risk.p_conf >= 0.25)) return "watch";
  return "calm";
}
const STATE_COLOR = { alert: C.red, watch: C.yellow, calm: C.green };

// ---- dashcam ----
function drawDashcam(f) {
  const cv = $("dashcam"), g = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const cam = state.data.scenario.camera;
  const proj = (X, Z) => [cam.cx + (cam.f * X) / Z, cam.y_h + (cam.f * cam.h) / Z];
  const yh = cam.y_h;

  // sky and ground
  let gr = g.createLinearGradient(0, 0, 0, yh);
  gr.addColorStop(0, "#3a4350"); gr.addColorStop(1, "#8b8f8f");
  g.fillStyle = gr; g.fillRect(0, 0, W, yh);
  gr = g.createLinearGradient(0, yh, 0, H);
  gr.addColorStop(0, "#5a5650"); gr.addColorStop(1, "#3c3a37");
  g.fillStyle = gr; g.fillRect(0, yh, W, H - yh);
  // unmarked road surface: darker asphalt band with dusty edges, no paint
  roadPoly(g, proj, -6.5, 8.5, 2.5, 120, "#2d2f31");
  roadPoly(g, proj, -5.8, 7.8, 2.5, 120, "#26282a");
  g.font = "16px Atkinson Hyperlegible, sans-serif";
  // horizon from calibration
  g.strokeStyle = "rgba(240,185,43,0.5)"; g.setLineDash([6, 6]); g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(0, yh); g.lineTo(W, yh); g.stroke(); g.setLineDash([]);
  g.fillStyle = "rgba(240,185,43,0.85)"; g.fillText(`horizon y_h = ${yh.toFixed(1)} px (fitted from traffic)`, 12, yh - 8);

  // the 1D baseline's fixed corridor
  const hw = state.data.scenario.corridor_half_width;
  const c = [proj(-hw, BLIND_ZONE), proj(hw, BLIND_ZONE), proj(hw, 45), proj(-hw, 45)];
  g.fillStyle = "rgba(79,138,224,0.14)"; g.strokeStyle = C.base; g.lineWidth = 2; g.setLineDash([10, 8]);
  g.beginPath(); c.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill(); g.stroke(); g.setLineDash([]);
  g.fillStyle = "#9cbcf0"; g.font = "bold 16px Atkinson Hyperlegible, sans-serif";
  const [lx, ly] = proj(hw + 0.2, 6);
  g.fillText("1D corridor", lx + 6, ly - 4);

  // actors, far to near
  const actors = [...f.actors].sort((a, b) => (b.truth[1] || 0) - (a.truth[1] || 0));
  for (const a of actors) {
    const st = actorState(a);
    const col = STATE_COLOR[st];
    let box = a.bbox, dashed = false;
    if (!box && a.coasting && a.track && a.track.pos[1] > 0.5) {
      const [w, h] = SIZE[a.cls]; const [X, Z] = a.track.pos;
      const [xc, yb] = proj(X, Z);
      box = [xc - (cam.f * w) / Z / 2, yb - (cam.f * h) / Z, xc + (cam.f * w) / Z / 2, yb];
      dashed = true;
    }
    if (!box) continue;
    const [x1, y1, x2, y2] = box;
    drawVehicle(g, a.cls, x1, y1, x2, y2);
    g.lineWidth = st === "alert" ? 4 : 2.5; g.strokeStyle = col;
    if (dashed) g.setLineDash([8, 6]);
    g.strokeRect(x1, y1, x2 - x1, y2 - y1); g.setLineDash([]);
    // label chip
    const ttc = a.risk && a.risk.ttc_2d !== null ? `TTC ${a.risk.ttc_2d.toFixed(1)} s` : "TTC ∞";
    const txt = `${LABEL[a.cls]} · ${a.risk ? ttc : "init"}${dashed ? " · blind zone, predicted" : ""}`;
    g.font = "bold 16px Atkinson Hyperlegible, sans-serif";
    const tw = g.measureText(txt).width + 14;
    const ty = Math.max(4, y1 - 26);
    g.fillStyle = "rgba(17,19,21,0.85)"; roundRect(g, x1, ty, tw, 22, 5); g.fill();
    g.fillStyle = col; g.fillRect(x1, ty, 4, 22);
    g.fillStyle = C.ink; g.fillText(txt, x1 + 9, ty + 16);
  }

  // ego hood and blind-zone note
  const [, yBZ] = proj(0, BLIND_ZONE);
  g.fillStyle = "#15171a";
  g.beginPath(); g.moveTo(0, H); g.lineTo(0, H - 30); g.quadraticCurveTo(W / 2, H - 70, W, H - 30); g.lineTo(W, H); g.fill();
  g.fillStyle = C.muted; g.font = "14px Atkinson Hyperlegible, sans-serif";
  g.fillText(`camera blind below ${BLIND_ZONE} m (bottom edge leaves the frame)`, 14, Math.min(H - 40, yBZ + 0) - 4);
}

function roadPoly(g, proj, xl, xr, z0, z1, fill) {
  const p = [proj(xl, z0), proj(xr, z0), proj(xr, z1), proj(xl, z1)];
  g.fillStyle = fill; g.beginPath(); p.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y))); g.closePath(); g.fill();
}

function drawVehicle(g, cls, x1, y1, x2, y2) {
  const w = x2 - x1, h = y2 - y1;
  g.save();
  if (cls === "auto_rickshaw") {
    g.fillStyle = "#1f6b3a"; roundRect(g, x1 + w * 0.05, y1 + h * 0.05, w * 0.9, h * 0.55, w * 0.25); g.fill();   // green canopy
    g.fillStyle = "#e3b324"; g.fillRect(x1 + w * 0.08, y1 + h * 0.55, w * 0.84, h * 0.35);                      // yellow body
    g.fillStyle = "#111"; g.fillRect(x1 + w * 0.15, y1 + h * 0.88, w * 0.18, h * 0.12); g.fillRect(x1 + w * 0.67, y1 + h * 0.88, w * 0.18, h * 0.12);
  } else if (cls === "two_wheeler" || cls === "bicycle") {
    g.fillStyle = "#c9ccd0"; g.beginPath(); g.ellipse(x1 + w / 2, y1 + h * 0.16, w * 0.28, h * 0.14, 0, 0, 7); g.fill();
    g.fillStyle = "#3d5a8a"; g.fillRect(x1 + w * 0.22, y1 + h * 0.28, w * 0.56, h * 0.42);
    g.fillStyle = "#111"; g.fillRect(x1 + w * 0.38, y1 + h * 0.7, w * 0.24, h * 0.3);
  } else if (cls === "truck" || cls === "bus") {
    g.fillStyle = cls === "bus" ? "#b8442f" : "#c46a1c"; g.fillRect(x1, y1, w, h * 0.92);
    g.fillStyle = "rgba(20,30,40,0.75)"; g.fillRect(x1 + w * 0.1, y1 + h * 0.1, w * 0.8, h * 0.3);
    g.fillStyle = "#111"; g.fillRect(x1 + w * 0.08, y1 + h * 0.9, w * 0.2, h * 0.1); g.fillRect(x1 + w * 0.72, y1 + h * 0.9, w * 0.2, h * 0.1);
  } else {
    g.fillStyle = "#9aa3ad"; roundRect(g, x1, y1 + h * 0.15, w, h * 0.75, w * 0.12); g.fill();
    g.fillStyle = "rgba(20,30,40,0.8)"; g.fillRect(x1 + w * 0.12, y1 + h * 0.22, w * 0.76, h * 0.25);
    g.fillStyle = "#111"; g.fillRect(x1 + w * 0.05, y1 + h * 0.88, w * 0.18, h * 0.12); g.fillRect(x1 + w * 0.77, y1 + h * 0.88, w * 0.18, h * 0.12);
  }
  g.restore();
}

function roundRect(g, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h / 2));
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

// ---- radar ----
function drawRadar(f) {
  const cv = $("radar"), g = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  const { fwdMax: FWD_MAX, fwdMin: FWD_MIN } = state.radar;
  const LAT = ((FWD_MAX - FWD_MIN) * W) / H / 2;
  const s = Math.min(W / (2 * LAT), H / (FWD_MAX - FWD_MIN));
  const P = (lat, fwd) => [W / 2 + lat * s, H - (fwd - FWD_MIN) * s];

  g.fillStyle = "#15181b"; g.fillRect(0, 0, W, H);
  // range rings
  const [ox, oy] = P(0, 0);
  g.strokeStyle = "rgba(185,190,195,0.13)"; g.lineWidth = 1; g.fillStyle = C.muted; g.font = "13px Atkinson Hyperlegible, sans-serif";
  const ring = FWD_MAX > 25 ? 10 : 5;
  for (let r = ring; r <= FWD_MAX; r += ring) {
    g.beginPath(); g.arc(ox, oy, r * s, Math.PI, 2 * Math.PI); g.stroke();
    g.fillText(`${r} m`, ox + 4, oy - r * s + 14);
  }
  g.beginPath(); g.moveTo(ox, 0); g.lineTo(ox, H); g.stroke();
  // corridor
  const hw = state.data.scenario.corridor_half_width;
  g.strokeStyle = C.base; g.setLineDash([8, 7]); g.lineWidth = 1.5;
  for (const sx of [-hw, hw]) { const [a1, b1] = P(sx, 0); const [a2, b2] = P(sx, FWD_MAX); g.beginPath(); g.moveTo(a1, b1); g.lineTo(a2, b2); g.stroke(); }
  g.setLineDash([]);
  g.fillStyle = "#9cbcf0"; g.fillText("1D corridor", P(hw, FWD_MAX * 0.9)[0] + 5, P(hw, FWD_MAX * 0.9)[1]);
  // ego truck
  const [tx, ty] = P(-1.25, 0);
  g.fillStyle = "#3b4045"; g.strokeStyle = C.yellow; g.lineWidth = 2;
  roundRect(g, tx, ty, 2.5 * s, 7.5 * s, 4); g.fill(); g.stroke();
  g.fillStyle = C.yellow; g.font = "bold 13px Atkinson Hyperlegible, sans-serif"; g.fillText("EGO TRUCK", tx - 2, ty + 7.5 * s + 16);
  g.strokeStyle = "rgba(240,185,43,0.45)"; g.setLineDash([4, 4]); g.beginPath(); g.arc(ox, oy, EGO_RADIUS * s, 0, 7); g.stroke(); g.setLineDash([]);

  const placed = [];
  const placeLabel = (x, y, w, h) => {   // nudge labels down until they stop overlapping
    let yy = y;
    for (let k = 0; k < 8; k++) {
      if (!placed.some((r) => x < r[0] + r[2] && x + w > r[0] && yy < r[1] + r[3] && yy + h > r[1])) break;
      yy += h + 2;
    }
    placed.push([x, yy, w, h]);
    return yy;
  };
  // draw cones first, then discs and labels on top
  const order = [...f.actors].sort((a, b) => (actorState(a) === "alert") - (actorState(b) === "alert"));
  for (const a of order) {
    const st = actorState(a), col = STATE_COLOR[st];
    if (a.paths) {
      // prediction cone: hull of every sigma-point future
      const pts = a.paths.flat();
      const hull = convexHull(pts);
      g.fillStyle = hexA(col, st === "alert" ? 0.26 : 0.15);
      g.strokeStyle = hexA(col, 0.65); g.lineWidth = 1.5;
      g.beginPath(); hull.forEach(([la, fw], i) => { const [x, y] = P(la, fw); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.closePath(); g.fill(); g.stroke();
      g.strokeStyle = hexA(col, 0.55); g.lineWidth = 1;
      for (const path of a.paths) { g.beginPath(); path.forEach(([la, fw], i) => { const [x, y] = P(la, fw); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); }
    }
  }
  for (const a of order) {
    const st = actorState(a), col = STATE_COLOR[st];
    if (a.track) {
      const [la, fw] = a.track.pos; const [x, y] = P(la, fw);
      const [maj, min, ang] = a.track.ellipse;
      g.strokeStyle = C.ink2; g.lineWidth = 1;
      g.beginPath(); g.ellipse(x, y, Math.max(1, maj * s), Math.max(1, min * s), -ang, 0, 7); g.stroke();
      g.fillStyle = col; g.strokeStyle = C.asphalt; g.lineWidth = 2;
      g.beginPath(); g.arc(x, y, RADIUS[a.cls] * s, 0, 7); g.fill(); g.stroke();
    }
    if ($("truth").checked && a.truth) {
      const [x, y] = P(...a.truth);
      g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x - 5, y - 5); g.lineTo(x + 5, y + 5); g.moveTo(x + 5, y - 5); g.lineTo(x - 5, y + 5); g.stroke();
    }
  }
  for (const a of order) {
    const col = STATE_COLOR[actorState(a)];
    if (a.track) {
      const [x, y] = P(...a.track.pos);
      const p = a.risk ? a.risk.p_conf.toFixed(2) : "–";
      const ttc = a.risk && a.risk.ttc_2d !== null ? `${a.risk.ttc_2d.toFixed(1)} s` : "∞";
      const label = `${LABEL[a.cls]}${a.tracked_as !== a.cls ? ` (as ${LABEL[a.tracked_as]})` : ""}  P ${p} · TTC ${ttc}`;
      g.font = "bold 13px Atkinson Hyperlegible, sans-serif";
      const tw = g.measureText(label).width;
      const lxp = Math.min(W - tw - 10, Math.max(6, x + RADIUS[a.cls] * s + 6));
      const ly = placeLabel(lxp - 4, y - 24, tw + 8, 19);
      g.fillStyle = "rgba(17,19,21,0.85)"; g.fillRect(lxp - 4, ly, tw + 8, 19);
      g.fillStyle = col; g.fillRect(lxp - 4, ly, 3, 19);
      g.fillStyle = C.ink; g.fillText(label, lxp + 2, ly + 14);
    }
  }
  // legend
  g.font = "13px Atkinson Hyperlegible, sans-serif";
  const items = [["alert", C.red], ["watch (P ≥ 0.25)", C.yellow], ["quiet", C.green]];
  items.forEach(([t, c], i) => { g.fillStyle = c; g.fillRect(12, 14 + i * 20, 12, 12); g.fillStyle = C.ink2; g.fillText(t, 30, 25 + i * 20); });
  g.fillStyle = C.muted; g.fillText("cone: 15 sigma-point futures relative to the ego's own path · ellipse: 2σ", 12, H - 12);
}

function convexHull(points) {
  const p = points.map((q) => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// ---- head-to-head lanes ----
function sizeLanes() {
  for (const id of ["lane-1d", "lane-2d"]) {
    const cv = $(id), r = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(200, Math.round(r.width * dpr)); cv.height = Math.round(56 * dpr);
  }
}
function drawLanes() {
  const d = state.data, sc = d.scenario, dur = sc.duration;
  const focus = d.summaries[sc.focus] || null;
  const lanes = [["lane-1d", "baseline"], ["lane-2d", "bharat"]];
  for (const [id, key] of lanes) {
    const cv = $(id), g = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
    const W = cv.width, H = cv.height;
    const X = (t) => (t / dur) * W;
    g.clearRect(0, 0, W, H);
    g.fillStyle = C.panel2; roundRect(g, 0, 0, W, H, 6 * dpr); g.fill();
    // quiet baseline line, then debounced alert spans
    g.fillStyle = hexA(C.green, 0.45); g.fillRect(0, H / 2 - 2 * dpr, W, 4 * dpr);
    const dt = 1 / sc.fps;
    g.fillStyle = C.red;
    for (const f of d.frames) if (f.actors.some((a) => a[key])) g.fillRect(X(f.t), H * 0.28, X(dt) + 1, H * 0.44);
    // 1 s grid
    g.fillStyle = "rgba(185,190,195,0.25)"; g.font = `${11 * dpr}px Atkinson Hyperlegible, sans-serif`;
    for (let s = 1; s < dur; s++) { g.fillRect(X(s), H - 6 * dpr, 1, 6 * dpr); if (id === "lane-1d") g.fillText(`${s}s`, X(s) + 2, H - 6 * dpr); }
    // corridor entry of the focus actor
    if (focus && focus.corridor_entry !== null) {
      g.strokeStyle = C.ink2; g.setLineDash([4 * dpr, 3 * dpr]); g.lineWidth = 1.5 * dpr;
      g.beginPath(); g.moveTo(X(focus.corridor_entry), 0); g.lineTo(X(focus.corridor_entry), H); g.stroke(); g.setLineDash([]);
    }
    // cue markers
    g.font = `bold ${12 * dpr}px Atkinson Hyperlegible, sans-serif`;
    for (const e of d.events.filter((e) => e.system === key)) {
      const x = X(e.t);
      g.fillStyle = C.ink; g.fillRect(x - dpr, 2 * dpr, 2 * dpr, H - 4 * dpr);
      const label = `${e.message} ${e.t.toFixed(2)} s`;
      const tw = g.measureText(label).width;
      const lx = x + 6 * dpr + tw > W ? x - tw - 6 * dpr : x + 6 * dpr;
      g.fillStyle = "rgba(17,19,21,0.85)"; g.fillRect(lx - 3 * dpr, 3 * dpr, tw + 6 * dpr, 16 * dpr);
      g.fillStyle = C.ink; g.fillText(label, lx, 15 * dpr);
    }
    // playhead
    g.fillStyle = C.yellow; g.fillRect(X(state.t) - 1.5 * dpr, 0, 3 * dpr, H);
  }
}

function renderKpis() {
  const d = state.data, sc = d.scenario, s = d.summaries[sc.focus];
  const fmt = (t) => (t === null || t === undefined ? "never" : `${t.toFixed(2)} s`);
  const cues = (sys) => d.events.filter((e) => e.system === sys).length;
  const items = [];
  if (s) {
    items.push([`Bharat-TTC fires · ${sc.focus}`, fmt(s.first_bharat), ""]);
    items.push([`1D baseline fires · ${sc.focus}`, fmt(s.first_baseline), ""]);
    if (s.lead_gain !== null) {
      const cls = s.lead_gain > 0 ? "good" : "bad";
      items.push(["Lead time", `${s.lead_gain > 0 ? "+" : ""}${s.lead_gain.toFixed(2)} s`, cls]);
      if (s.lead_gain > 0) items.push(["Extra braking room", `≈ ${Math.round(s.lead_gain * sc.ego_speed_kmh / 3.6)} m`, "good"]);
    }
    if (sc.t_conflict && s.first_baseline !== null) items.push(["Baseline margin", `${(sc.t_conflict - s.first_baseline).toFixed(2)} s`, "bad"]);
  }
  items.push(["Voice cues (2D)", `${cues("bharat")}`, ""]);
  items.push(["Beeps (1D)", `${cues("baseline")}`, ""]);
  $("kpis").innerHTML = items.map(([k, v, c]) => `<div class="kpi"><span class="k">${k}</span><span class="v ${c}">${v}</span></div>`).join("");
}

function renderEvents() {
  const ol = $("events");
  const ev = state.data.events;
  if (!ev.length) { ol.innerHTML = `<li class="empty">No alerts in this scenario.</li>`; return; }
  ol.innerHTML = ev.map((e) => `<li class="${e.system}" data-t="${e.t}"><span>${e.t.toFixed(2)} s</span>
    <span class="sys">${e.system === "bharat" ? "Bharat-TTC" : "1D baseline"}</span>
    <span class="msg">${e.system === "bharat" ? `“${e.message}”` : "beep · " + e.message} <small class="id">${e.id}</small></span></li>`).join("");
}

// ----------------------------------------------------------------- charts --
const playhead = {
  id: "playhead",
  afterDatasetsDraw(chart) {
    if (!state.data || chart.config.options.plugins.playhead === false) return;
    const x = chart.scales.x.getPixelForValue(state.t), { top, bottom } = chart.chartArea, g = chart.ctx;
    g.save(); g.strokeStyle = C.yellow; g.lineWidth = 2; g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke(); g.restore();
  },
};
const crosshair = {
  id: "crosshair",
  afterDraw(chart) {
    const a = chart.tooltip && chart.tooltip.getActiveElements();
    if (!a || !a.length) return;
    const x = a[0].element.x, { top, bottom } = chart.chartArea, g = chart.ctx;
    g.save(); g.strokeStyle = "rgba(238,240,241,0.35)"; g.lineWidth = 1; g.beginPath(); g.moveTo(x, top); g.lineTo(x, bottom); g.stroke(); g.restore();
  },
};

function baseOptions(yTitle, yMax) {
  return {
    responsive: true, maintainAspectRatio: false, animation: false, parsing: false, normalized: true,
    interaction: { mode: "index", intersect: false },
    scales: {
      x: { type: "linear", min: 0, max: state.data.scenario.duration, title: { display: true, text: "time (s)", color: C.muted },
           grid: { color: "rgba(185,190,195,0.08)" }, ticks: { color: C.muted }, border: { color: C.line } },
      y: { min: 0, max: yMax, title: { display: true, text: yTitle, color: C.muted },
           grid: { color: "rgba(185,190,195,0.08)" }, ticks: { color: C.muted }, border: { display: false } },
    },
    plugins: {
      legend: { labels: { color: C.ink2, boxWidth: 14, boxHeight: 3, font: { family: "Atkinson Hyperlegible" } } },
      tooltip: { backgroundColor: "rgba(17,19,21,0.95)", borderColor: C.line, borderWidth: 1, titleColor: C.ink, bodyColor: C.ink2,
                 callbacks: { title: (it) => `t = ${it[0].parsed.x.toFixed(2)} s`,
                              label: (it) => `${it.dataset.label}: ${it.parsed.y === null ? "∞" : it.parsed.y.toFixed(2)}` } },
    },
  };
}
const line = (label, data, color, extra = {}) => ({ label, data, borderColor: color, backgroundColor: color, borderWidth: 2,
  pointRadius: 0, pointHoverRadius: 4, spanGaps: false, ...extra });
const thresh = (label, y, color) => line(label, [{ x: 0, y }, { x: state.data.scenario.duration, y }], color,
  { borderDash: [6, 5], borderWidth: 1.5, pointHoverRadius: 0 });

function buildCharts() {
  for (const ch of Object.values(state.charts)) ch.destroy();
  state.charts = {};
  const series = (fn) => state.data.frames.map((f) => {
    const a = f.actors.find((x) => x.id === state.chartActor);
    return { x: f.t, y: a && a.risk ? fn(a.risk) : null };
  });
  const cap = (v) => (v === null ? null : Math.min(v, 6));
  state.charts.ttc = new Chart($("ttc-chart"), {
    type: "line",
    data: { datasets: [
      line("TTC₂D Bharat-TTC", series((r) => cap(r.ttc_2d)), C.bharat),
      line("TTC₁D baseline", series((r) => cap(r.ttc_1d)), C.base),
      thresh("2D threshold 1.8 s", 1.8, hexA(C.bharat, 0.7)),
      thresh("1D threshold 2.0 s", 2.0, hexA(C.base, 0.7)),
    ] },
    options: baseOptions("seconds (capped at 6)", 6),
    plugins: [playhead, crosshair],
  });
  state.charts.p = new Chart($("pconf-chart"), {
    type: "line",
    data: { datasets: [
      line("P_conf", series((r) => r.p_conf), C.bharat, { stepped: false }),
      thresh("alert threshold 0.5", 0.5, hexA(C.bharat, 0.7)),
    ] },
    options: baseOptions("probability", 1),
    plugins: [playhead, crosshair],
  });
}

function buildCalibration() {
  const cal = state.data.calibration;
  const W = { car: 1.75, auto_rickshaw: 1.35 };
  const pts = cal.samples.map(([yb, w, cls]) => ({ x: yb, y: w / W[cls] }));
  const xs = pts.map((p) => p.x), lo = Math.min(...xs), hi = Math.max(...xs);
  const fit = [{ x: lo, y: (lo - cal.y_h_fit) / cal.h_fit }, { x: hi, y: (hi - cal.y_h_fit) / cal.h_fit }];
  if (state.charts.calib) state.charts.calib.destroy();
  const opts = baseOptions("w / W_c  (px per m)", undefined);
  opts.scales.x = { type: "linear", title: { display: true, text: "bottom edge y_b (px)", color: C.muted },
                    grid: { color: "rgba(185,190,195,0.08)" }, ticks: { color: C.muted }, border: { color: C.line } };
  delete opts.scales.y.max;
  opts.interaction = { mode: "nearest", intersect: true };
  opts.plugins.playhead = false;
  opts.plugins.tooltip.callbacks = { title: () => "", label: (it) => `y_b ${it.parsed.x.toFixed(0)} px · w/W_c ${it.parsed.y.toFixed(1)}` };
  state.charts.calib = new Chart($("calib-chart"), {
    type: "scatter",
    data: { datasets: [
      { label: "car / auto boxes", data: pts, backgroundColor: hexA(C.ink2, 0.45), borderWidth: 0, pointRadius: 2.5, pointHoverRadius: 5 },
      { label: "Huber fit", type: "line", data: fit, borderColor: C.yellow, borderWidth: 2, pointRadius: 0 },
    ] },
    options: opts,
  });
  $("calib-stats").innerHTML =
    `Fit of w/W<sub>c</sub> = y<sub>b</sub>/h − y<sub>h</sub>/h over ${cal.n} boxes (${Math.round(cal.inlier_fraction * 100)}% inliers): ` +
    `<b>h = ${cal.h_fit.toFixed(3)} m</b> (true ${cal.h_true}), <b>y<sub>h</sub> = ${cal.y_h_fit.toFixed(1)} px</b> (true ${cal.y_h_true}). No lane markings used.`;
}

// ------------------------------------------------------------ custom form --
function setupCustomForm() {
  const form = $("custom-form");
  const out = { swing_deg: ["o-swing", "°"], ego_kmh: ["o-ego", " km/h"], auto_kmh: ["o-auto", " km/h"] };
  for (const [name, [id, unit]] of Object.entries(out)) {
    form.elements[name].addEventListener("input", (e) => { $(id).textContent = e.target.value + unit; });
  }
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const el = form.elements;
    const payload = {
      swing_deg: +el.swing_deg.value, ego_kmh: +el.ego_kmh.value, auto_kmh: +el.auto_kmh.value,
      tracked_as: el.tracked_as.value, use_crawl_gate: el.use_crawl_gate.checked, seed: +el.seed.value,
    };
    $("custom-status").textContent = "Running the pipeline…";
    $("run-custom").disabled = true;
    try {
      await loadScenario("custom", payload);
      const sel = $("scenario");
      let opt = [...sel.options].find((o) => o.value === "custom");
      if (!opt) { opt = new Option("Custom cut-in", "custom"); sel.add(opt); }
      opt.textContent = state.data.scenario.title;
      sel.value = "custom";
      const s = state.data.summaries["auto-1"];
      $("custom-status").textContent = s.lead_gain === null
        ? `Bharat-TTC ${s.first_bharat === null ? "did not fire" : "fired at " + s.first_bharat.toFixed(2) + " s"}; baseline ${s.first_baseline === null ? "did not fire" : "fired at " + s.first_baseline.toFixed(2) + " s"}.`
        : `Lead time ${s.lead_gain > 0 ? "+" : ""}${s.lead_gain.toFixed(2)} s over the 1D baseline. Press Play.`;
    } catch (err) {
      $("custom-status").textContent = `Failed: ${err}`;
    } finally {
      $("run-custom").disabled = false;
    }
  });
  // The scenario <select> also needs to reload "custom" from cache if picked again.
  $("scenario").addEventListener("change", (e) => {
    if (e.target.value === "custom") form.requestSubmit();
  }, true);
}

Chart.defaults.font.family = '"Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif';
Chart.defaults.color = C.muted;
if ("speechSynthesis" in window) window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
window.appReady = init();
