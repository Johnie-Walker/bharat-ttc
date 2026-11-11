"""Bharat-TTC judge dashboard: FastAPI backend serving the simulator and the static UI.

Run from the repo folder:

    .venv\\Scripts\\python.exe -m uvicorn web.server:app --port 8000    (Windows)
    .venv/bin/python -m uvicorn web.server:app --port 8000             (macOS/Linux)

then open http://127.0.0.1:8000
"""
from __future__ import annotations

import sys
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bharat_ttc import ActorClass, __version__, evidence  # noqa: E402
from bharat_ttc.sim import (  # noqa: E402
    collision_course_cut_in,
    result_to_json,
    run_scenario,
    scenario_library,
)

STATIC = Path(__file__).resolve().parent / "static"
_EVIDENCE: dict = {"data": None, "error": None}


def _disable_power_throttling() -> bool:
    """Opt this process out of Windows EcoQoS power throttling, as a real-time process would.

    Without it, Windows tends to run a background server on efficiency cores at
    low clocks, which inflates the Layer 1 timing several-fold. No-op elsewhere.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        class _State(ctypes.Structure):
            _fields_ = [("Version", ctypes.c_ulong), ("ControlMask", ctypes.c_ulong),
                        ("StateMask", ctypes.c_ulong)]

        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        st = _State(1, 0x1, 0x0)  # EXECUTION_SPEED controlled, state off = not throttled
        ok = k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(st), ctypes.sizeof(st))
        # Also ask for above-normal priority so the demo server is not starved by the browser.
        k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00008000)
        return bool(ok)
    except Exception:
        return False


def _warm() -> None:
    """Pre-compute every library scenario and the evidence so the demo never waits."""
    try:
        throttling_off = _disable_power_throttling()
        for key in scenario_library():
            _library_run(key)
        data = evidence.compute_all()
        data["benchmark"]["power_throttling_disabled"] = throttling_off
        _EVIDENCE["data"] = data
    except Exception as exc:  # surfaced through /api/evidence
        _EVIDENCE["error"] = repr(exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Bharat-TTC dashboard", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class CustomCutIn(BaseModel):
    swing_deg: float = Field(30.0, ge=5.0, le=50.0, description="cut-in swing angle (deg)")
    ego_kmh: float = Field(48.0, ge=20.0, le=80.0)
    auto_kmh: float = Field(20.0, ge=8.0, le=45.0)
    tracked_as: str = Field("auto_rickshaw", description="agility prior: auto_rickshaw or truck")
    use_crawl_gate: bool = True
    seed: int = Field(7, ge=0, le=10_000)


@lru_cache(maxsize=16)
def _library_run(key: str) -> dict:
    lib = scenario_library()
    if key not in lib:
        raise KeyError(key)
    return result_to_json(run_scenario(lib[key]))


@lru_cache(maxsize=64)
def _custom_run(p: tuple) -> dict:
    swing, ego, auto, tracked, gate, seed = p
    tracked_cls = ActorClass.parse(tracked)
    spec = collision_course_cut_in(
        swing, ego_kmh=ego, auto_kmh=auto,
        tracked_as=None if tracked_cls is ActorClass.AUTO_RICKSHAW else tracked_cls,
        use_crawl_gate=gate, seed=seed, key="custom",
        title=f"Custom: {swing:.0f}° cut-in, truck {ego:.0f} km/h, auto {auto:.0f} km/h"
              + ("" if tracked_cls is ActorClass.AUTO_RICKSHAW else f" (tracked as {tracked_cls.value})"),
    )
    return result_to_json(run_scenario(spec))


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    return [{"key": s.key, "title": s.title, "description": s.description}
            for s in scenario_library().values()]


@app.get("/api/simulate/{key}")
def simulate(key: str) -> dict:
    try:
        return _library_run(key)
    except KeyError:
        raise HTTPException(404, f"unknown scenario {key!r}") from None


@app.get("/api/evidence")
def get_evidence():
    if _EVIDENCE["error"]:
        raise HTTPException(500, _EVIDENCE["error"])
    if _EVIDENCE["data"] is None:
        return JSONResponse({"status": "computing"}, status_code=202)
    return _EVIDENCE["data"]


@app.post("/api/custom")
def custom(p: CustomCutIn) -> dict:
    try:
        ActorClass.parse(p.tracked_as)
    except ValueError:
        raise HTTPException(422, f"unknown class {p.tracked_as!r}") from None
    return _custom_run((round(p.swing_deg, 1), round(p.ego_kmh, 1), round(p.auto_kmh, 1),
                        p.tracked_as, p.use_crawl_gate, p.seed))
