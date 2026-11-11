"""Near-miss records: what Layer 1 sends to Layer 2 on every alert (deck, stage F).

The deck specifies "~300 bytes: H3 cell, time, speed, min TTC, deceleration,
other actor's class, lighting. No video by default." This module builds that
record from an alert and encodes it two ways, so the size is measured, not assumed:

  * compact JSON (readable, what a first integration would send)
  * packed binary (struct, fixed 32 bytes) for constrained uplinks

Severity uses DRAC ~ closing speed / TTC (deceleration rate to avoid the crash),
as in the deck's Layer 2 scoring.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict, dataclass

from .motion_models import ActorClass

SCHEMA_VERSION = 1
H3_RESOLUTION = 9           # ~0.1 km^2 cells, as in the deck
M_PER_DEG_LAT = 111_320.0

_CLASS_CODE = {c: i for i, c in enumerate(ActorClass)}
_SIDE_CODE = {"left": 0, "ahead": 1, "right": 2}
_LIGHT_CODE = {"day": 0, "dusk": 1, "night_lit": 2, "night_unlit": 3}
# version, h3 cell, unix time, ego speed cm/s, min TTC ms, ego accel cm/s^2,
# DRAC cm/s^2, P_conf x255, class, side, lighting, flags, 6 bytes device id
_PACK = struct.Struct("<BQIHHhHBBBBB6s")


def h3_cell(lat: float, lon: float, res: int = H3_RESOLUTION) -> str:
    try:
        import h3
    except ImportError:  # keep Layer 1 usable without the h3 wheel
        return f"latlon:{lat:.5f},{lon:.5f}"
    return h3.latlng_to_cell(lat, lon, res)


def local_to_latlon(x_east: float, y_north: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Flat-earth offset (metres) from an origin; fine over a few km."""
    lat = lat0 + y_north / M_PER_DEG_LAT
    lon = lon0 + x_east / (M_PER_DEG_LAT * math.cos(math.radians(lat0)))
    return lat, lon


def drac(closing_speed: float, ttc: float) -> float:
    """Deceleration rate to avoid the crash, ~ closing speed / TTC (m/s^2)."""
    if not math.isfinite(ttc) or ttc <= 0 or closing_speed <= 0:
        return 0.0
    return closing_speed / ttc


@dataclass
class NearMissRecord:
    v: int                 # schema version
    h3: str                # H3 res-9 cell of the ego at the alert
    t: int                 # unix time (s)
    ego_kmh: float
    min_ttc_s: float       # min TTC_2D seen on this track up to the alert
    ego_accel: float       # m/s^2 (negative = braking)
    drac: float            # m/s^2, severity
    p_conf: float
    actor: str             # ActorClass value
    side: str              # left / ahead / right
    light: str             # day / dusk / night_lit / night_unlit
    trigger: str = "alert"  # "alert" or "hard_brake"
    dev: str = "000000"    # pseudonymous device id (hex, 6 chars); no driver or plate data

    def to_json(self) -> str:
        d = asdict(self)
        for k in ("ego_kmh", "min_ttc_s", "ego_accel", "drac", "p_conf"):
            d[k] = round(d[k], 2)
        return json.dumps(d, separators=(",", ":"))

    def to_bytes(self) -> bytes:
        try:
            import h3
            cell = int(h3.str_to_int(self.h3))
        except Exception:
            cell = 0
        flags = 1 if self.trigger == "hard_brake" else 0
        return _PACK.pack(
            self.v, cell, self.t,
            _clamp(round(self.ego_kmh / 3.6 * 100), 0, 65535),
            _clamp(round(self.min_ttc_s * 1000), 0, 65535),
            _clamp(round(self.ego_accel * 100), -32768, 32767),
            _clamp(round(self.drac * 100), 0, 65535),
            _clamp(round(self.p_conf * 255), 0, 255),
            _CLASS_CODE[ActorClass(self.actor)], _SIDE_CODE[self.side], _LIGHT_CODE[self.light],
            flags, bytes.fromhex(self.dev.ljust(12, "0")[:12]),
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> "NearMissRecord":
        (v, cell, t, spd, ttc, acc, dr, p, c, s, light, flags, dev) = _PACK.unpack(b)
        try:
            import h3
            cell_s = h3.int_to_str(cell)
        except Exception:
            cell_s = str(cell)
        inv = lambda m, x: next(k for k, i in m.items() if i == x)  # noqa: E731
        return cls(v, cell_s, t, spd / 100 * 3.6, ttc / 1000, acc / 100, dr / 100, p / 255,
                   inv(_CLASS_CODE, c).value, inv(_SIDE_CODE, s), inv(_LIGHT_CODE, light),
                   "hard_brake" if flags & 1 else "alert", dev.hex()[:6])

    def sizes(self) -> dict:
        return {"json_bytes": len(self.to_json().encode()), "binary_bytes": len(self.to_bytes())}


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


def daily_upload_bytes(records_per_day: float, bytes_per_record: int) -> float:
    return records_per_day * bytes_per_record
