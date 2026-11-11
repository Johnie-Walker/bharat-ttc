---
marp: true
theme: default
paginate: true
math: mathjax
title: Bharat-TTC
description: Collision warning that works without lanes — Safer Roads for India
style: |
  @import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=Atkinson+Hyperlegible:wght@400;700&display=swap');
  section {
    font-family: "Atkinson Hyperlegible", "Segoe UI", sans-serif;
    font-size: 25px;
    background: #F3F4F1;
    color: #23272B;
    padding: 50px 64px;
  }
  h1, h2, h3 { font-family: "Archivo", "Arial Narrow", Arial, sans-serif; font-stretch: 85%; color: #23272B; }
  h2 { border-bottom: 5px solid #E2A100; padding-bottom: 0.15em; margin-bottom: 0.5em; }
  table { font-size: 0.78em; }
  th { background: #F6E7BF; }
  blockquote { border-left: 6px solid #E2A100; background: #FFFFFF; color: #23272B; padding: 0.5em 0.9em; font-weight: 700; }
  .small { display: block; font-size: 0.72em; line-height: 1.4; color: #5A6167; }
  section.lead .small { color: #C9CDD0; }
  section::after { color: #5A6167; }
  section.lead { background: #3B4045; color: #EEF0F1; }
  section.lead h1 { color: #E2A100; font-size: 1.9em; line-height: 1.15; }
  section.lead h2 { color: #EEF0F1; border-bottom-color: #E2A100; }
  section.lead strong { color: #E2A100; }
  section.lead::after { color: #C9CDD0; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Bharat-TTC: collision warning that works without lanes

An open-source, camera-only risk engine for Indian mixed traffic.

Warns a fleet driver early when a two-wheeler or auto-rickshaw **cuts in from the side**, stays quiet in crawling traffic, and pools every near-miss into a map of road stretches turning dangerous.

<span class="small">Safer Roads for India · Open Innovation · Primary track: collision-risk prediction and behaviour detection · Secondary track: commercial-fleet safety and hazard awareness</span>

<!--
Open with the one-line idea: Western collision warning assumes lanes; Indian traffic negotiates sideways.
-->

---

## Why fleets, and why now

- **4,87,707** road accidents and **1,77,175** deaths in India in 2024 (MoRTH)
- Two-wheeler riders are the largest victim group: **44.8%** of deaths in 2023, pedestrians **20.4%**
- Fleets already install dashcams, so they are where AI can act fastest
- Alerts work when drivers trust them: a Netradyne–IISc–Zoomcar pilot (~1 lakh trips) saw **14% fewer accidents** with voice alerts on
- The industry admits the gap: Netradyne limits external-camera features in India because false alerts erode trust
- MoRTH mandates lane-departure warning for 8+ passenger commercial vehicles from 2026, on roads that often have no lane markings

<!--
The lever is cheap: hardware is already in the cab. The blocker is trust, and trust is lost to false alerts.
-->

---

## The core problem: lane-based warning misses angular cut-ins

| What forward collision warning assumes | What Indian roads do |
|---|---|
| Danger arrives inside a narrow corridor straight ahead | Two-wheelers and autos swerve in from the side, toward a median gap or into a space |
| An actor counts once it is in front of the bumper | By then there is the least braking time left |
| 1D TTC = distance ÷ closing speed | It shrinks whenever traffic bunches at low speed, so it fires all day in jams |

> Forward collision warning on Indian roads misses sideways cut-ins by two-wheelers and auto-rickshaws and over-alerts in slow traffic, so fleet drivers stop trusting it.

<span class="small">Affected: drivers of trucks, buses and cabs, and the two-wheeler riders, auto passengers and pedestrians around them.</span>

<!--
Two failure modes, one outcome: the driver mutes the system. Missed cut-ins hurt the largest victim group; false alerts in jams destroy trust.
-->

---

## The working prototype: one pipeline, two layers

| | Stage | Where | What it does |
|---|---|---|---|
| A | Detect and track Indian road users | Dashcam | Apache-2.0 detector, ONNX INT8; ByteTrack / BoT-SORT |
| B | Calibrate without lane markings | Dashcam | Camera height and horizon from vehicle sizes |
| C | Predict motion with class-specific agility | Dashcam | CTRV unscented Kalman filter per actor |
| D | Score conflict risk in 2D | Dashcam | Closed-form 2D TTC plus conflict probability |
| E | Decide when to speak | Dashcam | Speed- and direction-aware alert policy |
| F | Turn every near-miss into a map | Cloud | ~300-byte records pooled into an H3 risk map |

**Layer 1** (A–E) protects the driver in real time. **Layer 2** (F) pools what Layer 1 emits.

<!--
Everything safety-critical runs on the device. Only small event records leave the vehicle.
-->

---

## A–B: Detect, track and calibrate from the traffic itself

**A.** RT-DETR-R18 or D-FINE-S (Apache-2.0, no AGPL obligations for vendors). Classes: person, rider/two-wheeler, bicycle, auto-rickshaw, car, bus, truck, animal. Camera-motion compensation keeps IDs stable over potholes and speed breakers.

**B.** Most monocular calibration needs painted lanes. On a flat road, a vehicle's pixel width grows linearly with how low it sits in the image:

$$ w = \frac{W_c}{h}\,(y_b - y_h) $$

A robust fit over auto-rickshaw and car detections gives horizon $y_h$ and camera height $h$. Then:

$$ Z = \frac{f\,h}{y_b - y_h}, \qquad X = \frac{(x - c_x)\,h}{y_b - y_h}, \qquad \sigma_Z \approx \frac{Z^2\,\sigma_y}{f\,h} $$

Range error grows with distance squared, so it feeds the tracker as measurement noise; alerts stay within ~35 m.

<!--
Calibration re-runs continuously as a self-check; the device flags itself if the camera is knocked out of position.
-->

---

## C–D: Class-specific motion, scored in 2D

**C.** Each actor gets a CTRV unscented Kalman filter with state $\mathbf{x} = [X, Y, v, \psi, \dot\psi]^\top$. Process noise depends on class: an auto or two-wheeler gets a wide prediction cone, a bus a narrow one. Variances $\sigma^2_{a,c}$, $\sigma^2_{\ddot\psi,c}$ are estimated from Indian traffic video, not hand-picked.

**D.** With $\mathbf{r} = \mathbf{p}_i - \mathbf{p}_{\text{ego}}$, $\mathbf{v} = \mathbf{v}_i - \mathbf{v}_{\text{ego}}$ and combined safety radius $R$ (defined when $\mathbf{r}\cdot\mathbf{v} < 0$):

$$ \mathrm{TTC}_{2D} = \frac{-\,\mathbf{r}\cdot\mathbf{v} - \sqrt{(\mathbf{r}\cdot\mathbf{v})^2 - \lVert\mathbf{v}\rVert^2(\lVert\mathbf{r}\rVert^2 - R^2)}}{\lVert\mathbf{v}\rVert^2} $$

Sigma points rolled forward 2.5 s give the chance the actor's path meets the ego's:

$$ P_{\text{conf}} = \sum_j w_j\, \mathbb{1}\!\left[\min_{t \le 2.5\,\mathrm{s}} \lVert \mathbf{p}^{(j)}_i(t) - \mathbf{p}_{\text{ego}}(t) \rVert < R \right] $$

<!--
TTC alone ignores uncertainty; P_conf carries the class-specific cone into the decision.
-->

---

## E: Decide when to speak

- **Crawl mode:** ego below ~18 km/h and lateral closing below ~1.2 m/s → longitudinal alerts suppressed. Bumper-to-bumper bunching is normal, not a hazard.
- **Cut-in alert:** $P_{\text{conf}}$ high, $\mathrm{TTC}_{2D}$ under ~1.8 s, and the actor's lateral motion converging on the ego path
- **Debounce:** two of three consecutive frames, plus a per-actor cooldown
- **One short voice cue that names the side:** *"auto, left"*, so the driver knows where to look

Thresholds are starting points, tuned on validation data to **maximise cut-in recall at a fixed budget of false alerts per hour**.

<!--
This is where trust is won: silence in jams, one directional cue when it matters.
-->

---

## F: Turn every near-miss into a map (Layer 2)

- On every alert, or a GPS hard brake (≤ about −3 m/s²), the device sends **~300 bytes**: H3 cell, time, speed, min TTC, deceleration, other actor's class, lighting. No video by default.
- Records are map-matched to OpenStreetMap and scored per H3 resolution-9 cell (~0.1 km²). Severity uses DRAC ≈ closing speed ÷ TTC, weighted up for vulnerable road users and decayed over time.
- Shrinkage toward the city average, by fleet exposure $E_h$, stops one unlucky pass painting a cell red:

$$ \hat\lambda_h = \frac{\alpha + \sum_{k \in h} w_k\, s_k\, e^{-\lambda (t_{\text{now}} - t_k)}}{\beta + E_h} $$

- **Grey-spots**: high-scoring stretches with no blackspot record, ranked by a gradient-boosted model on OSM features and explained with SHAP. Presented as items for site inspection.

<span class="small">An official blackspot needs 5 serious accidents or 10 deaths within ~500 m over three years. MoRTH's 2021 circular asks agencies to act before that threshold.</span>

<!--
Example explanation: "most conflicts after 6 pm, unlit, near an unsignalised median cut."
-->

---

## How we will prove it works

**Fair baseline:** same detector, tracker and calibration; only the risk logic is swapped for lane-based 1D TTC < 2 s in a fixed forward corridor. **Ablations:** constant-velocity vs CTRV, shared vs class-specific agility, crawl gate on vs off.

| Hypothesis | Target |
|---|---|
| H1: Earlier warning on angular cut-ins | Median gain ≥ 1.0 s |
| H2: Fewer nuisance alerts in slow traffic, equal or better recall | ≥ 50% fewer alerts/hour below 20 km/h |
| H3: Runs on hardware fleets already buy | ≥ 15 fps on Jetson Orin Nano class |
| H4: Near-miss map is stable, not noise | Split-half rank correlation ≥ 0.5 |
| H5: Cheap at fleet scale | < 1 MB upload per vehicle per day |

<span class="small">Evaluated on ~150–200 METEOR cut-in events (CC BY-NC-SA, evaluation only, no shipped weights). Map built from ThirdEye Delhi NCR GPS logs (CC BY 4.0). All figures are targets to be measured, not results.</span>

<!--
At 50 km/h, every extra second of warning is roughly 14 m more room to brake or yield.
-->

---

<!-- _class: lead -->

# Western collision warning assumes lanes. Indian traffic negotiates sideways.

Two-wheeler riders are almost half of India's road deaths, and many die when a vehicle meets them at an angle that lane-based warning never watches. That same warning beeps all day in a jam, so drivers switch it off.

**Bharat-TTC** tracks every road user in 2D with motion models learned from Indian traffic. It aims to speak up **a second or more earlier** when an auto or bike cuts in, and to **stay quiet at crawl speed**.

Every warning becomes a 300-byte near-miss record, showing road authorities which stretches are filling up with near-misses **before** they reach the deaths an official blackspot requires.

**Built for the road India actually has.**

<!--
Close on the deployment path: open Apache-2.0 release, 4–6 week shadow mode with a partner fleet, then a driver-facing A/B pilot.
-->
