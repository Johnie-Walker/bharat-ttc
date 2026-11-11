"""Class-specific CTRV motion models (Layer 1, stage C).

State on the ground plane: x = [X, Y, v, psi, psi_dot]
    X, Y     world position (m)
    v        speed along the heading (m/s)
    psi      heading, radians anticlockwise from +X
    psi_dot  yaw rate (rad/s)

Process noise is two white inputs held constant over a step:
    nu_a      longitudinal acceleration, variance sigma_a^2
    nu_psidd  yaw acceleration,          variance sigma_psidd^2
Both variances depend on the vehicle class, which is what makes an
auto-rickshaw's predicted future spread into a wide cone and a bus's stay narrow.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np


class ActorClass(str, Enum):
    PERSON = "person"
    TWO_WHEELER = "two_wheeler"
    BICYCLE = "bicycle"
    AUTO_RICKSHAW = "auto_rickshaw"
    CAR = "car"
    BUS = "bus"
    TRUCK = "truck"
    ANIMAL = "animal"

    @classmethod
    def parse(cls, name: "str | ActorClass") -> "ActorClass":
        """Accept detector-style labels: 'auto-rickshaw', 'rider', 'motorcycle', ..."""
        if isinstance(name, ActorClass):
            return name
        key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
        return cls(_CLASS_ALIASES.get(key, key))

    @property
    def spoken(self) -> str:
        """Word used in the driver's voice cue ("auto, left")."""
        return _SPOKEN[self]


_CLASS_ALIASES = {
    "auto": "auto_rickshaw", "autorickshaw": "auto_rickshaw", "rickshaw": "auto_rickshaw",
    "rider": "two_wheeler", "motorcycle": "two_wheeler", "motorbike": "two_wheeler",
    "scooter": "two_wheeler", "bike": "two_wheeler", "pedestrian": "person",
    "cycle": "bicycle", "lorry": "truck", "cow": "animal", "dog": "animal",
}
_SPOKEN = {
    ActorClass.PERSON: "Pedestrian", ActorClass.TWO_WHEELER: "Bike",
    ActorClass.BICYCLE: "Cycle", ActorClass.AUTO_RICKSHAW: "Auto",
    ActorClass.CAR: "Car", ActorClass.BUS: "Bus", ActorClass.TRUCK: "Truck",
    ActorClass.ANIMAL: "Animal",
}


@dataclass(frozen=True)
class ClassAgility:
    sigma_a: float       # std of longitudinal-acceleration noise (m/s^2)
    sigma_psidd: float   # std of yaw-acceleration noise (rad/s^2)
    yaw_rate_max: float  # physical yaw-rate cap applied when rolling futures forward (rad/s)
    v_max: float         # speed cap applied when rolling futures forward (m/s)
    radius: float        # footprint radius, summed with the ego's to give R (m)

    @property
    def Q(self) -> np.ndarray:
        return np.diag([self.sigma_a**2, self.sigma_psidd**2])


# Starting priors only. The document's plan is to replace these with
# estimate_agility() run on tracked METEOR trajectories, one call per class.
# Auto-rickshaw / rider and bus / truck use the hackathon brief's priors:
#   auto, rider: sigma_a = 2.6 m/s^2, sigma_psidd = 1.2 rad/s^2
#   bus, truck:  sigma_a = 1.2 m/s^2, sigma_psidd = 0.25 rad/s^2
CLASS_AGILITY: dict[ActorClass, ClassAgility] = {
    ActorClass.PERSON:        ClassAgility(0.8, 1.5, 2.0, 3.0, 0.5),
    ActorClass.TWO_WHEELER:   ClassAgility(2.6, 1.2, 1.0, 22.0, 0.8),
    ActorClass.BICYCLE:       ClassAgility(1.0, 1.0, 1.0, 8.0, 0.8),
    ActorClass.AUTO_RICKSHAW: ClassAgility(2.6, 1.2, 0.9, 16.0, 1.1),
    ActorClass.CAR:           ClassAgility(2.0, 0.5, 0.6, 33.0, 1.5),
    ActorClass.BUS:           ClassAgility(1.2, 0.25, 0.35, 25.0, 2.0),
    ActorClass.TRUCK:         ClassAgility(1.2, 0.25, 0.3, 25.0, 2.0),
    ActorClass.ANIMAL:        ClassAgility(0.5, 1.0, 1.0, 4.0, 1.0),
}

YAW_RATE_EPS = 1e-4  # |psi_dot| at or below this is integrated as a straight line


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def noise_gain(psi: float, dt: float) -> np.ndarray:
    """G (5x2) mapping the step-constant inputs [nu_a, nu_psidd] into the state.

    Same terms ctrv_step adds for nu, so the additive process noise of the
    filter is Q = G diag(sigma_a^2, sigma_psidd^2) G^T. It depends on heading,
    so it must be rebuilt every prediction step.
    """
    half = 0.5 * dt * dt
    return np.array([
        [half * np.cos(psi), 0.0],
        [half * np.sin(psi), 0.0],
        [dt,                 0.0],
        [0.0,                half],
        [0.0,                dt],
    ])


def process_noise(agility: ClassAgility, psi: float, dt: float) -> np.ndarray:
    """Class-specific CTRV process noise covariance Q (5x5) for one step of dt."""
    G = noise_gain(psi, dt)
    return G @ agility.Q @ G.T


def ctrv_step(x: np.ndarray, dt: float, nu: np.ndarray | None = None) -> np.ndarray:
    """Propagate CTRV state(s) by dt. Works on (5,) or (..., 5) arrays.

    Turning (|psi_dot| > 1e-4):
        X' = X + v/psi_dot (sin(psi + psi_dot dt) - sin psi)
        Y' = Y + v/psi_dot (-cos(psi + psi_dot dt) + cos psi)
    Straight (|psi_dot| <= 1e-4):
        X' = X + v cos(psi) dt,  Y' = Y + v sin(psi) dt
    Both: v' = v, psi' = psi + psi_dot dt, psi_dot' = psi_dot.

    nu, if given, is (..., 2) = [nu_a, nu_psidd] held constant over the step.
    """
    x = np.asarray(x, dtype=float)
    v, psi, w = x[..., 2], x[..., 3], x[..., 4]
    straight = np.abs(w) <= YAW_RATE_EPS
    psi_new = psi + w * dt
    c, s = np.cos(psi), np.sin(psi)
    r = v / np.where(straight, 1.0, w)
    out = np.empty_like(x)
    out[..., 0] = x[..., 0] + np.where(straight, v * c * dt, r * (np.sin(psi_new) - s))
    out[..., 1] = x[..., 1] + np.where(straight, v * s * dt, r * (c - np.cos(psi_new)))
    out[..., 2] = v
    out[..., 3] = psi_new
    out[..., 4] = w
    if nu is not None:
        nu = np.asarray(nu, dtype=float)
        na, nd = nu[..., 0], nu[..., 1]
        half = 0.5 * dt * dt
        out[..., 0] += half * c * na
        out[..., 1] += half * s * na
        out[..., 2] += dt * na
        out[..., 3] += half * nd
        out[..., 4] += dt * nd
    return out


@dataclass
class EgoState:
    """Ego pose from GPS/IMU odometry, in the same world frame as the tracks."""
    x: float
    y: float
    v: float
    psi: float
    yaw_rate: float = 0.0

    @property
    def state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.v, self.psi, self.yaw_rate])

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])

    @property
    def velocity(self) -> np.ndarray:
        return self.v * np.array([np.cos(self.psi), np.sin(self.psi)])

    def rollout(self, dt: float, n_steps: int) -> np.ndarray:
        """Noise-free CTRV positions at t = 0, dt, ..., n_steps*dt. Shape (n_steps+1, 2).

        Closed form: chaining exact CTRV steps at constant v and yaw rate gives
        the same positions as one CTRV step of length t, so no loop is needed.
        """
        t = np.arange(n_steps + 1) * dt
        w, v, psi = self.yaw_rate, self.v, self.psi
        if abs(w) <= YAW_RATE_EPS:
            dx, dy = v * np.cos(psi) * t, v * np.sin(psi) * t
        else:
            dx = v / w * (np.sin(psi + w * t) - np.sin(psi))
            dy = v / w * (np.cos(psi) - np.cos(psi + w * t))
        return np.column_stack([self.x + dx, self.y + dy])


def _robust_std(a: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def estimate_agility(
    tracks: Iterable[np.ndarray], dt: float, min_speed: float = 1.0
) -> tuple[float, float]:
    """Estimate (sigma_a, sigma_psidd) for one class from smoothed ground-plane tracks.

    Each track is an (N, 2) array of positions sampled every dt seconds, already
    smoothed (e.g. by an RTS pass of this filter). Samples slower than min_speed
    are dropped because heading is meaningless there. A median-based spread is
    used so occasional ID switches or glitches don't inflate the result.
    Treat the output as starting values and confirm with filter consistency (NIS).
    """
    accels, yaw_accels = [], []
    for p in tracks:
        step = np.diff(np.asarray(p, dtype=float), axis=0)
        speed = np.hypot(step[:, 0], step[:, 1]) / dt
        heading = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
        a = np.diff(speed) / dt
        yaw_rate = np.diff(heading) / dt
        yaw_acc = np.diff(yaw_rate) / dt
        moving = speed > min_speed
        accels.append(a[moving[:-1] & moving[1:]])
        yaw_accels.append(yaw_acc[moving[:-2] & moving[1:-1] & moving[2:]])
    return _robust_std(np.concatenate(accels)), _robust_std(np.concatenate(yaw_accels))
