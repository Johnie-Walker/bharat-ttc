# Bharat-TTC — prototype demo runbook

For whoever presents the prototype to the judges. Target length: about 3 minutes, then questions.

## Before the session

1. Windows: double-click `run_dashboard.bat` (first run creates `.venv` and installs requirements: about 5 minutes and needs internet, so do it before the session, not in front of the judges).
   macOS/Linux: follow the manual steps in the README. The browser opens at http://127.0.0.1:8000.
2. Wait about 10 s after the server starts before opening **Evidence** (the server pre-computes 40+ runs).
3. Turn the laptop volume up: the voice cue ("Auto, left!") is part of the demo.
4. Plug in the charger. The server opts out of Windows power throttling, but on battery the CPU can still clock down and make the
   compute benchmark slower (the Evidence tab reports the spread either way).
5. Works offline: Chart.js is bundled; fonts fall back to system fonts.

## The 3-minute walkthrough

Click **★ Guided judge tour**. It pauses at each moment below; press **Next**. The tour's numbers are read
from the running simulation, so they always match the screen.

| Step | What the judges see | What to say | Criterion |
|---|---|---|---|
| 1 | Dashcam, no lane markings | "Most systems need painted lanes. We calibrate the camera from the traffic itself: car and auto widths." | Innovation |
| 2 | "Auto, left!" spoken; red cone hits the truck on the radar | "The auto is still 4–5 m to the side, outside the lane corridor. We already know it's coming." | Technical, UX |
| 3 | Baseline beeps 1.27 s later | "Same detections, same tracker, only the risk logic differs. That's ~17 m more braking room at 48 km/h (1.27 s lead over the baseline × 13.3 m/s)." | Technical |
| 4 | Jam: baseline beeps, Bharat-TTC silent | "This is why drivers switch warnings off. Our crawl gate knows bunching isn't a hazard." | Business (trust) |
| 5 | Auto squeezes in, cue fires | "But a real sideways squeeze-in still gets through." | Functionality |
| 6 | Evidence: 40-run sweep | "Not one cherry-picked clip. Median +1.17 s. Where it's weaker is on the chart too." | Technical |
| 7 | Hypothesis scorecard | "Each deck target, scored honestly: met in simulation, directional, or still to measure." | Presentation |
| 8 | Fleet: 204-byte record | "Every alert leaves the truck as ~200 bytes, no video. Pooled per H3 cell it maps near-misses before they become blackspots." | Business |
| 9 | "Try your own cut-in" | Invite a judge to pick an angle and speed. | UX |

## Likely questions, honest answers

- **Is this real footage?** No. Every number comes from synthetic scenarios with fixed seeds. The pipeline
  starts from bounding boxes, so a real detector's output can be fed in; METEOR evaluation is the next step.
- **Does class-specific agility make the warning earlier?** Not on the cut-in: truck agility warns at the same
  time. The lead comes from the 2D geometry. Agility makes swerve tracking ~2.5× more accurate.
- **Where does it lose?** Shallow drifts under ~15° (the gain drops below 1 s), and a swerve that straightens
  out ahead (the baseline is 0.8 s earlier). Both are shown on the Evidence tab.
- **Does it run in real time?** Layer 1 takes about 1–2 ms per tracked actor per frame on a laptop CPU in pure
  Python. The detector and a Jetson have not been measured yet (H3 is open).
- **What leaves the vehicle?** Only the near-miss record: H3 cell, time, speed, min TTC, deceleration, actor
  class, lighting, DRAC, P_conf, a pseudonymous device id. No video, no plate or driver data.

## If something goes wrong

- **No voice:** the red banner still shows the cue. Some browsers need one click on the page first.
- **Evidence says "computing":** wait a few seconds; it fills in automatically.
- **Port 8000 busy:** run `.venv\Scripts\python.exe -m uvicorn web.server:app --port 8010` from the repo folder
  (macOS/Linux: `.venv/bin/python -m uvicorn web.server:app --port 8010`).
- **Terminal-only backup:** `.venv\Scripts\python.exe demo_cut_in.py` (macOS/Linux: `.venv/bin/python demo_cut_in.py`) prints the same timeline and evidence.
