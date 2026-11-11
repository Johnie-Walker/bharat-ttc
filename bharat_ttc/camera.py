"""Monocular ground-plane measurement model and lane-free calibration (stage B).

Calibration without lane markings. On a flat road a vehicle of real width W_c
whose bottom edge sits on image row y_b has pixel width

    w = (W_c / h) (y_b - y_h)    =>    w / W_c = (1/h) y_b - y_h / h

so a robust straight-line fit of w/W_c against y_b over ordinary traffic
(cars W_c = 1.75 m, auto-rickshaws W_c = 1.35 m) gives slope 1/h and
intercept -y_h/h: camera height h and horizon row y_h, without the focal
length and without a single painted line. Huber or RANSAC regression keeps
misclassified trucks and half-occluded boxes from dragging the fit.

Back-projection of a detection's bottom-centre pixel then gives a ground
position with a distance-dependent covariance:

    Z = f h / (y_b - y_h),   X = (x - c_x) h / (y_b - y_h),   sigma_Z ~ Z^2 sigma_y / (f h)

The camera frame is ego-relative: X lateral (right positive), Z forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .motion_models import ActorClass, EgoState

# Nominal real widths used as the calibration ruler (from the brief).
CALIBRATION_WIDTH: dict[ActorClass, float] = {
    ActorClass.CAR: 1.75,
    ActorClass.AUTO_RICKSHAW: 1.35,
}

# (width, height) in metres for drawing synthetic boxes. Only the widths of
# CALIBRATION_WIDTH classes are trusted by the calibration.
VEHICLE_SIZE: dict[ActorClass, tuple[float, float]] = {
    ActorClass.PERSON: (0.5, 1.65),
    ActorClass.TWO_WHEELER: (0.75, 1.55),
    ActorClass.BICYCLE: (0.6, 1.6),
    ActorClass.AUTO_RICKSHAW: (1.35, 1.75),
    ActorClass.CAR: (1.75, 1.5),
    ActorClass.BUS: (2.6, 3.2),
    ActorClass.TRUCK: (2.5, 3.3),
    ActorClass.ANIMAL: (0.7, 1.4),
}


@dataclass(frozen=True)
class CameraModel:
    f: float              # focal length (px)
    cx: float             # principal-point column (px)
    y_h: float            # horizon row (px)
    h: float              # camera height above the road (m)
    sigma_x: float = 1.5  # pixel noise of the bottom-centre column
    sigma_y: float = 1.5  # pixel noise of the bottom-edge row

    def project(self, X: float, Z: float) -> tuple[float, float]:
        """Ground point (lateral X, forward Z) -> bottom-centre pixel (x, y_b)."""
        return self.cx + self.f * X / Z, self.y_h + self.f * self.h / Z

    def backproject(self, x: float, y_b: float) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> ([X, Z], 2x2 covariance), by first-order error propagation."""
        d = y_b - self.y_h
        if d <= 0:
            raise ValueError("bottom edge is at or above the horizon; no ground intersection")
        Z = self.f * self.h / d
        X = (x - self.cx) * self.h / d
        J = np.array([
            [self.h / d, -(x - self.cx) * self.h / d**2],
            [0.0,        -self.f * self.h / d**2],
        ])
        R = J @ np.diag([self.sigma_x**2, self.sigma_y**2]) @ J.T
        return np.array([X, Z]), R

    def pixel_to_ground(self, x: float, y_b: float) -> tuple[float, float]:
        """Bottom-centre pixel -> ground (X lateral, Z forward) in metres."""
        d = y_b - self.y_h
        if d <= 0:
            raise ValueError("bottom edge is at or above the horizon; no ground intersection")
        return (x - self.cx) * self.h / d, self.f * self.h / d

    def sigma_Z(self, Z: float) -> float:
        """Range standard deviation sigma_Z = Z^2 sigma_y / (f h)."""
        return Z**2 * self.sigma_y / (self.f * self.h)

    def bbox(self, X: float, Z: float, width: float, height: float) -> np.ndarray:
        """Ideal box [x1, y1, x2, y2] (px) of a width x height object standing at (X, Z)."""
        xc, yb = self.project(X, Z)
        half_w = 0.5 * self.f * width / Z
        return np.array([xc - half_w, yb - self.f * height / Z, xc + half_w, yb])

    def backproject_bbox(self, box: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Detector box -> ([X, Z], covariance), using its bottom-centre pixel."""
        x_b, y_b = bbox_bottom_centre(box)
        return self.backproject(x_b, y_b)

    @classmethod
    def from_calibration(cls, cal: "Calibration", f: float, cx: float,
                         sigma_x: float = 1.5, sigma_y: float = 1.5) -> "CameraModel":
        return cls(f=f, cx=cx, y_h=cal.y_h, h=cal.h, sigma_x=sigma_x, sigma_y=sigma_y)


def bbox_bottom_centre(box: Sequence[float]) -> tuple[float, float]:
    x1, _, x2, y2 = box
    return 0.5 * (x1 + x2), float(y2)


# ---- Lane-free calibration -----------------------------------------------

@dataclass
class Calibration:
    h: float                 # camera height (m)
    y_h: float               # horizon row (px)
    slope: float             # fitted 1/h
    intercept: float         # fitted -y_h/h
    inliers: np.ndarray      # bool mask over the input detections
    residual_std: float      # robust spread of w/W_c about the line (1/m)
    method: str
    n: int = field(default=0)

    @property
    def inlier_fraction(self) -> float:
        return float(self.inliers.mean()) if self.inliers.size else 0.0


def calibrate_from_vehicles(
    y_b: Sequence[float],
    width_px: Sequence[float],
    real_width: Sequence[float],
    method: str = "huber",
    huber_epsilon: float = 1.35,
    ransac_threshold: float | None = None,
    random_state: int = 0,
) -> Calibration:
    """Fit camera height h and horizon y_h from vehicle boxes, no lane markings needed.

    Regresses t = w / W_c on y_b:  t = (1/h) y_b - y_h/h.

    method: "huber" (sklearn HuberRegressor) or "ransac" (RANSACRegressor around
    LinearRegression). y_b is standardised before fitting for conditioning and
    the coefficients are mapped back afterwards.
    """
    from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor

    yb = np.asarray(y_b, dtype=float)
    t = np.asarray(width_px, dtype=float) / np.asarray(real_width, dtype=float)
    if yb.size < 3:
        raise ValueError("need at least 3 detections to calibrate")
    mu, sd = yb.mean(), yb.std() or 1.0
    u = ((yb - mu) / sd)[:, None]

    if method == "huber":
        # Scale the target too so the Huber threshold epsilon is in robust-std units.
        t_scale = np.median(np.abs(t - np.median(t))) or 1.0
        reg = HuberRegressor(epsilon=huber_epsilon, alpha=0.0, max_iter=1000)
        reg.fit(u, t / t_scale)
        a_u, b_u = reg.coef_[0] * t_scale, reg.intercept_ * t_scale
        # Huber's own outliers_ flags everything past epsilon (its linear zone),
        # which is most of the clean data too; use a 3-sigma robust band instead.
        inliers = None
    elif method == "ransac":
        if ransac_threshold is None:
            # Size the consensus band from a robust (Huber) pre-fit; residuals of
            # plain least squares are inflated by the very outliers RANSAC must reject.
            pre = calibrate_from_vehicles(yb, width_px, real_width, method="huber",
                                          huber_epsilon=huber_epsilon)
            r0 = t - (pre.slope * yb + pre.intercept)
            ransac_threshold = 2.5 * 1.4826 * np.median(np.abs(r0 - np.median(r0)))
        thr = ransac_threshold
        reg = RANSACRegressor(LinearRegression(), residual_threshold=thr,
                              random_state=random_state, max_trials=500)
        reg.fit(u, t)
        a_u, b_u = reg.estimator_.coef_[0], reg.estimator_.intercept_
        inliers = reg.inlier_mask_
    else:
        raise ValueError(f"unknown method {method!r}; use 'huber' or 'ransac'")

    slope = a_u / sd
    intercept = b_u - a_u * mu / sd
    if slope <= 0:
        raise ValueError("fitted slope is not positive; boxes do not grow toward the bottom of the image")
    resid = t - (slope * yb + intercept)
    if inliers is None:
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        inliers = np.abs(resid) <= 3.0 * scale
    r_in = resid[inliers]
    return Calibration(
        h=1.0 / slope,
        y_h=-intercept / slope,
        slope=slope,
        intercept=intercept,
        inliers=np.asarray(inliers, dtype=bool),
        residual_std=float(1.4826 * np.median(np.abs(r_in - np.median(r_in)))),
        method=method,
        n=int(yb.size),
    )


def calibrate_from_detections(
    detections: Sequence[tuple[ActorClass | str, Sequence[float]]], method: str = "huber"
) -> Calibration:
    """Convenience wrapper: [(class, [x1, y1, x2, y2]), ...] -> Calibration.

    Only classes with a known nominal width (car, auto-rickshaw) are used.
    """
    yb, w, W = [], [], []
    for cls, box in detections:
        c = ActorClass.parse(cls)
        if c not in CALIBRATION_WIDTH:
            continue
        x1, _, x2, y2 = box
        yb.append(y2)
        w.append(x2 - x1)
        W.append(CALIBRATION_WIDTH[c])
    return calibrate_from_vehicles(yb, w, W, method=method)


def simulate_calibration_traffic(
    cam: CameraModel,
    n: int,
    rng: np.random.Generator,
    outlier_fraction: float = 0.15,
    px_noise: float = 1.5,
    z_range: tuple[float, float] = (6.0, 45.0),
    image_width: int = 1280,
    image_height: int = 720,
) -> list[tuple[ActorClass, np.ndarray]]:
    """Synthetic background traffic boxes (cars and autos) seen by a true camera.

    Real-world contamination is injected as outliers: trucks the detector
    mislabels as cars (too wide) and partially occluded boxes (too narrow).
    """
    out: list[tuple[ActorClass, np.ndarray]] = []
    while len(out) < n:
        cls = ActorClass.CAR if rng.random() < 0.55 else ActorClass.AUTO_RICKSHAW
        Z = rng.uniform(*z_range)
        X = rng.uniform(-7.0, 7.0)
        W, Hh = VEHICLE_SIZE[cls]
        # Real vehicles vary a little around the nominal width.
        W_true = W * rng.normal(1.0, 0.04)
        if rng.random() < outlier_fraction:
            W_true = W * (rng.uniform(1.35, 1.6) if rng.random() < 0.5 else rng.uniform(0.45, 0.7))
        box = cam.bbox(X, Z, W_true, Hh) + rng.normal(0.0, px_noise, 4)
        if box[0] < 0 or box[2] > image_width or box[3] > image_height:
            continue
        out.append((cls, box))
    return out


def _ego_axes(ego: EgoState) -> np.ndarray:
    """Columns are the ego's right and forward unit vectors in the world frame."""
    c, s = np.cos(ego.psi), np.sin(ego.psi)
    return np.array([[s, c], [-c, s]])


def camera_to_world(z_cam: np.ndarray, R_cam: np.ndarray, ego: EgoState) -> tuple[np.ndarray, np.ndarray]:
    M = _ego_axes(ego)
    return ego.position + M @ z_cam, M @ R_cam @ M.T


def world_to_camera(p: np.ndarray, ego: EgoState) -> np.ndarray:
    return _ego_axes(ego).T @ (np.asarray(p, dtype=float) - ego.position)


def rotate_to_camera(vec: np.ndarray, ego: EgoState) -> np.ndarray:
    """Express a world-frame vector (e.g. relative velocity) as [lateral, forward]."""
    return _ego_axes(ego).T @ np.asarray(vec, dtype=float)
