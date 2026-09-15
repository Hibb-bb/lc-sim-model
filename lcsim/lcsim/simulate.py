"""Synthetic variable-star light curves with band-dependent behaviour, and an
observation model for surveys of different quality.

Latents per star (all stored so that they can be probed later):
    cls      class id in {0..4}: sinusoid, rrlyrae, eclipsing, doublemode, drw
    logP     log10 period in days (timescale for the aperiodic class)
    amp      intrinsic amplitude at the reference wavelength (550 nm)
    phase    phase offset in [0,1)  -- a nuisance, never probed
    color    "temperature" in [0,1]: sets how amplitude scales with wavelength
    fine     amplitude of a narrow periodic bump, relative to amp  -- the
             feature that only a low-noise, dense observation can resolve
    shape    one class-specific shape parameter (eclipse width, skew, ...)

Tiers by design:
    coarse : cls, logP, amp(band)         recoverable from any observation
    fine   : fine                          needs low noise + dense cadence
    color  : color                         needs two bands (control latent)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

CLASSES = ["sinusoid", "rrlyrae", "eclipsing", "doublemode", "drw"]
LAMBDA_REF = 550.0  # nm


@dataclass
class SurveyConfig:
    name: str
    wavelength_nm: float
    sigma: float           # per-point photometric noise (flux units)
    keep_frac: float       # fraction of grid points observed
    bad_prob: float = 0.0  # probability that a given observation is a "bad night"
    bad_sigma_mult: float = 5.0
    bad_keep_mult: float = 0.4
    smooth_days: float = 0.0   # boxcar integration time (long-cadence / long-exposure survey)


@dataclass
class SimConfig:
    n_grid: int = 512
    window_days: float = 30.0
    logP_range: tuple = (0.0, 0.78)      # 1 .. 6 days
    amp_range: tuple = (0.2, 1.0)
    fine_range: tuple = (0.0, 0.25)
    bump_width: float = 0.06             # in phase units
    bump_phase: float = 0.72
    color_gamma: float = 1.5             # amp(l) = amp * (l_ref/l)^(gamma*color)
    color_lag: float = 0.04              # phase lag per unit color per (l-l_ref)/l_ref
    class_probs: tuple | None = None     # e.g. (0.05, 0.1, 0.75, 0.05, 0.05): EW-like dominance of eclipsing
    seed: int = 0


# --------------------------------------------------------------------------- #
# latents
# --------------------------------------------------------------------------- #

def sample_latents(n: int, cfg: SimConfig, rng: np.random.Generator) -> dict:
    if cfg.class_probs is None:
        cls = rng.integers(0, len(CLASSES), size=n)
    else:
        pr = np.asarray(cfg.class_probs, dtype=float); pr = pr / pr.sum()
        cls = rng.choice(len(CLASSES), size=n, p=pr)
    logP = rng.uniform(*cfg.logP_range, size=n)
    amp = rng.uniform(*cfg.amp_range, size=n)
    phase = rng.uniform(0, 1, size=n)
    color = rng.uniform(0, 1, size=n)
    fine = rng.uniform(*cfg.fine_range, size=n)
    shape = rng.uniform(0, 1, size=n)
    return dict(cls=cls, logP=logP, amp=amp, phase=phase, color=color, fine=fine, shape=shape)


# --------------------------------------------------------------------------- #
# base shapes, all with roughly unit peak-to-peak and zero mean
# --------------------------------------------------------------------------- #

def _sinusoid(ph, shape):
    return 0.5 * np.sin(2 * np.pi * ph)


def _rrlyrae(ph, shape):
    # fast rise over `rise` of the phase, slow decline, skew set by shape
    rise = 0.10 + 0.15 * shape
    x = np.mod(ph, 1.0)
    y = np.where(x < rise, -0.5 + x / rise, 0.5 - (x - rise) / (1 - rise))
    # soften the corners a little so it is not a pure sawtooth
    return y - 0.08 * np.sin(4 * np.pi * x)


def _eclipsing(ph, shape):
    w = 0.04 + 0.06 * shape                       # eclipse half-width in phase
    x = np.mod(ph + 0.5, 1.0) - 0.5               # primary at phase 0
    prim = np.exp(-0.5 * (x / w) ** 2)
    x2 = np.mod(ph, 1.0) - 0.5                    # secondary at phase 0.5
    sec = 0.45 * np.exp(-0.5 * (x2 / w) ** 2)
    y = -(prim + sec)
    return y - y.mean(axis=-1, keepdims=True)


def _doublemode(ph, shape):
    ratio = 0.72 + 0.06 * shape                   # second period / first period
    return 0.35 * np.sin(2 * np.pi * ph) + 0.2 * np.sin(2 * np.pi * ph / ratio + 1.0)


def _drw(t_days, tau_days, amp, rng):
    """Damped random walk (OU process) on the grid, unit-variance then scaled."""
    n, m = t_days.shape
    dt = t_days[:, 1:] - t_days[:, :-1]
    rho = np.exp(-dt / tau_days[:, None])
    y = np.empty((n, m))
    y[:, 0] = rng.standard_normal(n)
    eps = rng.standard_normal((n, m - 1))
    for k in range(1, m):
        y[:, k] = rho[:, k - 1] * y[:, k - 1] + np.sqrt(1 - rho[:, k - 1] ** 2) * eps[:, k - 1]
    y = y / (y.std(axis=1, keepdims=True) + 1e-8) * 0.35
    return y - y.mean(axis=1, keepdims=True)


def _bump(ph, width, center):
    x = np.mod(ph - center + 0.5, 1.0) - 0.5
    return np.exp(-0.5 * (x / width) ** 2)


# --------------------------------------------------------------------------- #
# flux at a given wavelength
# --------------------------------------------------------------------------- #

def flux(latents: dict, t_days: np.ndarray, wavelength_nm: float, cfg: SimConfig,
         rng: np.random.Generator) -> np.ndarray:
    """Noise-free flux for every star at its own time grid.

    t_days: (n, m) times in days. Returns (n, m).
    """
    n, m = t_days.shape
    P = 10.0 ** latents["logP"]
    amp_band = latents["amp"] * (LAMBDA_REF / wavelength_nm) ** (cfg.color_gamma * latents["color"])
    lag = cfg.color_lag * latents["color"] * (wavelength_nm - LAMBDA_REF) / LAMBDA_REF
    ph = t_days / P[:, None] + latents["phase"][:, None] + lag[:, None]

    y = np.zeros((n, m))
    cls = latents["cls"]
    shape = latents["shape"][:, None]
    for k, fn in enumerate([_sinusoid, _rrlyrae, _eclipsing, _doublemode]):
        idx = cls == k
        if idx.any():
            y[idx] = fn(ph[idx], shape[idx])
    idx = cls == 4
    if idx.any():
        y[idx] = _drw(t_days[idx], P[idx], amp_band[idx], rng)

    # the fine feature: a narrow periodic bump, present in every class
    y = y + latents["fine"][:, None] * _bump(ph, cfg.bump_width, cfg.bump_phase)
    return amp_band[:, None] * y


# --------------------------------------------------------------------------- #
# observation model
# --------------------------------------------------------------------------- #

def observe(latents: dict, survey: SurveyConfig, cfg: SimConfig, rng: np.random.Generator,
            t0: np.ndarray | None = None) -> dict:
    """Observe every star once with `survey`.

    Returns a dict of arrays:
        x       (n, 4, n_grid) channels: flux (0 where missing), mask, sigma, wavelength
        q       (n,) quality score = log(n_points) - log(median sigma)
        is_bad  (n,) whether this observation was a bad night
        amp_band (n,) the band-specific amplitude (a probe target)
    """
    n = len(latents["cls"])
    m = cfg.n_grid
    if t0 is None:
        t0 = rng.uniform(0, 1000.0, size=n)      # independent epochs per observation
    grid = np.linspace(0, cfg.window_days, m, endpoint=False)
    t = t0[:, None] + grid[None, :]

    y = flux(latents, t, survey.wavelength_nm, cfg, rng)
    if survey.smooth_days > 0:
        k = max(1, int(round(survey.smooth_days / (cfg.window_days / m))))
        if k > 1:  # boxcar along time: the survey integrates over `smooth_days`
            pad = np.pad(y, ((0, 0), (k // 2, k - 1 - k // 2)), mode="edge")
            cs = np.cumsum(pad, axis=1)
            cs = np.concatenate([np.zeros((n, 1)), cs], axis=1)
            y = (cs[:, k:] - cs[:, :-k]) / k

    is_bad = rng.uniform(size=n) < survey.bad_prob
    sigma = np.where(is_bad, survey.sigma * survey.bad_sigma_mult, survey.sigma)
    keep = np.where(is_bad, survey.keep_frac * survey.bad_keep_mult, survey.keep_frac)
    # mild night-to-night scatter in the noise so sigma is informative but not constant
    sigma = sigma * rng.uniform(0.8, 1.2, size=n)

    mask = rng.uniform(size=(n, m)) < keep[:, None]
    # make sure nothing is completely empty
    empty = ~mask.any(axis=1)
    if empty.any():
        mask[empty, rng.integers(0, m, size=empty.sum())] = True

    sig_pt = sigma[:, None] * rng.uniform(0.7, 1.3, size=(n, m))
    y_obs = (y + rng.standard_normal((n, m)) * sig_pt) * mask
    sig_ch = sig_pt * mask
    wl_ch = np.full((n, m), (survey.wavelength_nm - LAMBDA_REF) / LAMBDA_REF)

    x = np.stack([y_obs, mask.astype(float), sig_ch, wl_ch], axis=1).astype(np.float32)
    npts = mask.sum(axis=1)
    med_sig = np.array([np.median(s[mk]) for s, mk in zip(sig_pt, mask)])
    q = np.log(npts) - np.log(med_sig)

    amp_band = latents["amp"] * (LAMBDA_REF / survey.wavelength_nm) ** (cfg.color_gamma * latents["color"])
    return dict(x=x, q=q.astype(np.float32), is_bad=is_bad, amp_band=amp_band.astype(np.float32))


def make_dataset(n: int, surveys: list[SurveyConfig], cfg: SimConfig, seed: int) -> dict:
    """Latents + one observation per survey for every star."""
    rng = np.random.default_rng(seed)
    lat = sample_latents(n, cfg, rng)
    obs = {s.name: observe(lat, s, cfg, rng) for s in surveys}
    return dict(latents=lat, obs=obs)
