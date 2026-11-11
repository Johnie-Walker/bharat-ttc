"""CTRV unscented Kalman filter with class-specific agility (Layer 1, stage C).

Built on ``filterpy.kalman.UnscentedKalmanFilter`` with Merwe scaled sigma
points. State x = [X, Y, v, psi, psi_dot]; the measurement is the ground-plane
position [X, Y] from the camera back-projection.

Process noise is what makes the filter class-aware. Each actor's two white
inputs (longitudinal acceleration nu_a, yaw acceleration nu_psidd) have
class-specific standard deviations, e.g.

    auto-rickshaw / rider: sigma_a = 2.6 m/s^2, sigma_psidd = 1.2 rad/s^2
    bus / truck:           sigma_a = 1.2 m/s^2, sigma_psidd = 0.25 rad/s^2

and enter the state through the CTRV noise gain G(psi, dt), so before every
prediction sub-step the filter's Q is rebuilt as

    Q = G(psi, dt) diag(sigma_a^2, sigma_psidd^2) G(psi, dt)^T

which follows the current heading (acceleration noise acts along the direction
of travel, not along a fixed axis).

For conflict scoring (risk.py) the filter also exposes *augmented* sigma
points over [x, nu_a, nu_psidd], so each rolled-out future can hold one
manoeuvre (keep braking, keep tightening the turn) for the whole horizon.
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from .motion_models import (
    CLASS_AGILITY,
    ActorClass,
    ClassAgility,
    ctrv_step,
    process_noise,
    wrap_angle,
)

N_X = 5
N_Z = 2
N_AUG = 7
PSI = 3
H = np.eye(N_Z, N_X)


def _sqrtm_psd(A: np.ndarray) -> np.ndarray:
    """Upper factor U with U^T U = A (filterpy's convention); falls back to eigh if A is semi-definite."""
    try:
        return np.linalg.cholesky(A).T
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(0.5 * (A + A.T))
        return (vecs @ np.diag(np.sqrt(np.clip(vals, 0.0, None)))).T


def residual_x(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d[..., PSI] = wrap_angle(d[..., PSI])
    return d


def state_mean(sigmas: np.ndarray, Wm: np.ndarray) -> np.ndarray:
    """Weighted mean with a circular mean for the heading."""
    x = Wm @ sigmas
    x[PSI] = np.arctan2(Wm @ np.sin(sigmas[:, PSI]), Wm @ np.cos(sigmas[:, PSI]))
    return x


def _hx(x: np.ndarray) -> np.ndarray:
    return x[:N_Z]


def _fx(x: np.ndarray, dt: float) -> np.ndarray:
    return ctrv_step(x, dt)


class _VectorisedUKF(UnscentedKalmanFilter):
    """filterpy's UKF with the process step applied to all sigma points in one call.

    The stock compute_process_sigmas loops over sigma points in Python; ctrv_step
    is vectorised, so this gives identical sigmas_f for a fraction of the cost.
    Everything else (unscented transform, update, weights) is filterpy's own.
    """

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        self.sigmas_f = ctrv_step(sigmas, dt)


def merwe_points(n: int, alpha: float, beta: float, kappa: float) -> MerweScaledSigmaPoints:
    return MerweScaledSigmaPoints(n, alpha=alpha, beta=beta, kappa=kappa,
                                  sqrt_method=_sqrtm_psd, subtract=residual_x if n == N_X else None)


class CTRVUKF:
    """One tracked actor: filterpy UKF with class-specific, heading-aware Q.

    alpha, beta, kappa are the Merwe scaling parameters of the filter's own
    sigma points. The default (alpha=1, beta=2, kappa=3-n) spreads points at
    sqrt(3) sigma, which suits CTRV's strong heading nonlinearity better than
    the tiny alpha often quoted for near-linear problems.
    """

    def __init__(
        self,
        actor_class: ActorClass | str,
        x0: np.ndarray,
        P0: np.ndarray,
        agility: ClassAgility | None = None,
        max_step: float = 0.1,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 3.0 - N_X,
    ):
        self.actor_class = ActorClass.parse(actor_class)
        self.agility = agility or CLASS_AGILITY[self.actor_class]
        self.max_step = max_step
        self.step_dt = max_step  # length of the step the per-step noise Q applies to
        self.kf = _VectorisedUKF(
            dim_x=N_X, dim_z=N_Z, dt=max_step, hx=_hx, fx=_fx,
            points=merwe_points(N_X, alpha, beta, kappa),
            x_mean_fn=state_mean, residual_x=residual_x,
        )
        self.kf.x = np.asarray(x0, dtype=float).copy()
        self.kf.P = np.asarray(P0, dtype=float).copy()
        self.kf.Q = process_noise(self.agility, self.kf.x[PSI], max_step)
        self.nis = float("nan")

    # ---- construction ----------------------------------------------------

    @classmethod
    def from_detection(
        cls,
        actor_class: ActorClass | str,
        z: np.ndarray,
        R: np.ndarray,
        heading: float,
        speed: float = 0.0,
        heading_std: float = np.pi / 4,
        speed_std: float | None = None,
        agility: ClassAgility | None = None,
    ) -> CTRVUKF:
        """Start a track from one ground-plane detection.

        Speed and heading aren't observable from a single frame, so they start
        from priors (typically the ego's heading, since most actors travel with
        the road) with wide variances scaled to the class's physical limits.
        """
        ag = agility or CLASS_AGILITY[ActorClass.parse(actor_class)]
        if speed_std is None:
            speed_std = ag.v_max / 2
        x0 = np.array([z[0], z[1], speed, heading, 0.0])
        P0 = np.diag([0.0, 0.0, speed_std**2, heading_std**2, (ag.yaw_rate_max / 2) ** 2])
        P0[:2, :2] = R
        return cls(actor_class, x0, P0, agility=ag)

    # ---- state access ----------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self.kf.x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self.kf.x = np.asarray(value, dtype=float).copy()

    @property
    def P(self) -> np.ndarray:
        return self.kf.P

    @P.setter
    def P(self, value: np.ndarray) -> None:
        self.kf.P = np.asarray(value, dtype=float).copy()

    @property
    def Q(self) -> np.ndarray:
        """Process noise the last prediction sub-step used."""
        return self.kf.Q

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2] * np.array([np.cos(self.x[PSI]), np.sin(self.x[PSI])])

    # ---- filter ----------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Predict dt ahead in sub-steps of at most max_step (CTRV is stiff in heading)."""
        n_sub = max(1, int(np.ceil(dt / self.max_step)))
        self.step_dt = dt / n_sub
        for _ in range(n_sub):
            self.kf.Q = process_noise(self.agility, self.kf.x[PSI], self.step_dt)
            self.kf.predict(dt=self.step_dt)
            self._normalise()

    def update(self, z: np.ndarray, R: np.ndarray) -> float:
        """UKF update with a ground-plane position; returns the NIS (chi^2, 2 dof)."""
        self.kf.update(np.asarray(z, dtype=float), R=np.asarray(R, dtype=float))
        self._normalise()
        y = self.kf.y
        self.nis = float(y @ self.kf.SI @ y)
        return self.nis

    def _normalise(self) -> None:
        """Keep speed non-negative and heading wrapped.

        A track born with the ego's heading as prior explains oncoming traffic
        with a negative speed. (v, psi) and (-v, psi + pi) are the same motion,
        so flip to the positive-speed form, transforming P with
        J = diag(1, 1, -1, 1, 1) (the heading shift by pi is a constant).
        """
        if self.kf.x[2] < 0.0:
            self.kf.x[2] = -self.kf.x[2]
            self.kf.x[PSI] += np.pi
            J = np.diag([1.0, 1.0, -1.0, 1.0, 1.0])
            self.kf.P = J @ self.kf.P @ J.T
        self.kf.x[PSI] = wrap_angle(self.kf.x[PSI])
        self.kf.P = 0.5 * (self.kf.P + self.kf.P.T)

    # ---- sigma points for conflict scoring --------------------------------

    def augmented(self, noise_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Mean and covariance over [x, nu_a, nu_psidd]; noise variance scaled by noise_scale."""
        x_aug = np.concatenate([self.x, np.zeros(2)])
        P_aug = np.zeros((N_AUG, N_AUG))
        P_aug[:N_X, :N_X] = self.P
        P_aug[N_X:, N_X:] = noise_scale * self.agility.Q
        return x_aug, P_aug

    def augmented_sigma_points(
        self, lam: float = 1.0, noise_scale: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Merwe sigma points (15, 7) and mean weights over the augmented state.

        With alpha = 1 the Merwe scaling reduces to lambda = kappa, so lam is
        passed as kappa. lam >= 0 keeps every weight non-negative, which the
        conflict probability needs to be a valid probability.
        """
        x_aug, P_aug = self.augmented(noise_scale)
        pts = merwe_points(N_AUG, alpha=1.0, beta=0.0, kappa=lam)
        return pts.sigma_points(x_aug, P_aug), pts.Wm.copy()
