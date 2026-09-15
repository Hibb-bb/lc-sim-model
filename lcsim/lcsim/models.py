"""Encoder, predictor, SIGReg, augmentations and the loss for every variant."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #

class Encoder(nn.Module):
    """1D CNN over the fixed grid. Input (B, 4, L): flux, mask, sigma, wavelength."""

    def __init__(self, d_out: int = 64, width: int = 64):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv1d(4, w, 7, stride=2, padding=3), nn.BatchNorm1d(w), nn.GELU(),
            nn.Conv1d(w, w, 5, stride=2, padding=2), nn.BatchNorm1d(w), nn.GELU(),
            nn.Conv1d(w, 2 * w, 5, stride=2, padding=2), nn.BatchNorm1d(2 * w), nn.GELU(),
            nn.Conv1d(2 * w, 2 * w, 3, stride=2, padding=1), nn.BatchNorm1d(2 * w), nn.GELU(),
            nn.Conv1d(2 * w, 2 * w, 3, stride=1, padding=1), nn.BatchNorm1d(2 * w), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(4 * w, 4 * w), nn.GELU(),
            nn.Linear(4 * w, d_out),
        )

    def forward(self, x):
        h = self.net(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        return self.head(h)


class Predictor(nn.Module):
    """Cross-survey predictor conditioned on (source wavelength, target wavelength).

    One module covers every ordered pair of surveys; a residual MLP keeps it
    close to the identity when source and target bands coincide.
    """

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 2, hidden), nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, z, wl_src, wl_tgt):
        c = torch.stack([wl_src, wl_tgt], dim=1)
        return z + self.net(torch.cat([z, c], dim=1))


# --------------------------------------------------------------------------- #
# SIGReg: sketched isotropic-Gaussian regularizer (Epps–Pulley on random 1-D
# projections), as in LeJEPA.
# --------------------------------------------------------------------------- #

def sigreg(z: torch.Tensor, n_proj: int = 256, n_t: int = 17, t_max: float = 5.0) -> torch.Tensor:
    B, d = z.shape
    u = torch.randn(d, n_proj, device=z.device)
    u = u / u.norm(dim=0, keepdim=True)
    p = z @ u                                            # (B, n_proj)
    t = torch.linspace(-t_max, t_max, n_t, device=z.device)
    arg = p.unsqueeze(-1) * t                            # (B, n_proj, n_t)
    re = torch.cos(arg).mean(0)                          # empirical characteristic function
    im = torch.sin(arg).mean(0)
    target = torch.exp(-0.5 * t ** 2)                    # cf of N(0,1)
    w = target / math.sqrt(2 * math.pi)                  # N(0,1) density as weight
    dt = t[1] - t[0]
    stat = (((re - target) ** 2 + im ** 2) * w).sum(-1) * dt   # (n_proj,)
    return stat.mean()                                   # ~0.15 when collapsed, ~1/B at N(0,I)


# --------------------------------------------------------------------------- #
# augmentations (operate on (B, 4, L) tensors; never touch the wavelength ch.)
# --------------------------------------------------------------------------- #

def _finish(x, m, noise_max):
    """Apply point dropout already folded into m, then noise at or below the point's own error."""
    B, C, L = x.shape
    flux, sig = x[:, 0], x[:, 2]
    empty = m.sum(1) == 0
    if empty.any():
        m[empty] = x[empty, 1]
    beta = torch.rand(B, 1, device=x.device) * noise_max
    flux = flux + torch.randn_like(flux) * sig * beta
    out = x.clone()
    out[:, 0] = flux * m
    out[:, 1] = m
    out[:, 2] = sig * m
    return out


def augment(x: torch.Tensor, crop_min: float = 0.6, drop_max: float = 0.3, noise_max: float = 0.7):
    """One view: random contiguous crop (as masking) + point dropout + noise."""
    B, C, L = x.shape
    mask = x[:, 1]
    frac = torch.empty(B, device=x.device).uniform_(crop_min, 1.0)
    length = (frac * L).long()
    start = (torch.rand(B, device=x.device) * (L - length + 1)).long()
    ar = torch.arange(L, device=x.device)[None, :]
    inwin = (ar >= start[:, None]) & (ar < (start + length)[:, None])
    p = torch.rand(B, 1, device=x.device) * drop_max
    keep = torch.rand(B, L, device=x.device) >= p
    return _finish(x, mask * inwin * keep, noise_max)


def two_views(x: torch.Tensor, disjoint: bool = True, sub_min: float = 0.7, drop_max: float = 0.3,
              noise_max: float = 0.7):
    """Two views of the same observation. With disjoint=True they come from
    non-overlapping halves of the window (split point in [0.35, 0.65]), so they
    share the star but not the realization; each half is then sub-cropped,
    point-dropped and noised as in `augment`."""
    if not disjoint:
        return augment(x), augment(x)
    B, C, L = x.shape
    mask = x[:, 1]
    ar = torch.arange(L, device=x.device)[None, :]
    split = (torch.empty(B, device=x.device).uniform_(0.35, 0.65) * L).long()
    left = ar < split[:, None]
    swap = torch.rand(B, device=x.device) < 0.5
    w1 = torch.where(swap[:, None], ~left, left)
    w2 = ~w1
    views = []
    for w in (w1, w2):
        n = w.sum(1)
        frac = torch.empty(B, device=x.device).uniform_(sub_min, 1.0)
        length = (frac * n).long()
        first = torch.argmax(w.float(), dim=1)                # first index of the window
        start = first + (torch.rand(B, device=x.device) * (n - length + 1)).long()
        inwin = (ar >= start[:, None]) & (ar < (start + length)[:, None])
        p = torch.rand(B, 1, device=x.device) * drop_max
        keep = torch.rand(B, L, device=x.device) >= p
        views.append(_finish(x, mask * inwin * keep, noise_max))
    return views[0], views[1]


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #

def mse(a, b):
    return ((a - b) ** 2).mean()


def nt_xent_multi(z: torch.Tensor, star_id: torch.Tensor, temperature: float = 0.2):
    """Multi-positive InfoNCE: every other view of the same star is a positive."""
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (star_id[:, None] == star_id[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob * pos).sum(1).div(pos.sum(1)).mean()


class Method(nn.Module):
    """Wraps encoder (+ predictor) and computes the loss for one training variant.

    variants:
        lejepa       symmetric alignment of all views on full z + SIGReg
        contrastive  multi-positive NT-Xent over all views
        lejepa_aa / contrastive_aa    same loss, A paired with a 2nd independent A observation (good+good)
        lejepa_aab / contrastive_aab  same loss on A, A2 and B together (good+good+bad): does adding B hurt?
        aug_only     LeJEPA on augmentations of the survey-A observation only
        single_good  LeJEPA on two independent survey-A observations of each star
                     (caller passes the second A observation as xb) -- private-info ceiling
        ours         split z=[z_s|z_p]; L_aug on z; gated asymmetric predictor loss on z_s
        split_only   split, but symmetric cross loss on z_s
        gate_only    gated asymmetric predictor loss on full z, no split
    """

    def __init__(self, variant: str, d: int = 64, d_s: int = 32, lam: float = 1.0,
                 tau: float = 0.5, temperature: float = 0.2, width: int = 64, disjoint: bool = True):
        super().__init__()
        self.disjoint = disjoint
        self.variant, self.d, self.d_s, self.lam, self.tau, self.temperature = variant, d, d_s, lam, tau, temperature
        self.enc = Encoder(d, width)
        dp = d_s if variant in ("ours", "split_only") else d
        self.pred = Predictor(dp) if variant in ("ours", "gate_only") else None

    def slice(self, z):
        if self.variant in ("ours", "split_only"):
            return z[:, : self.d_s]
        return z

    def cross_gated(self, zs_a, zs_b, q_a, q_b, wl_a, wl_b):
        w_ba = torch.sigmoid((q_a - q_b) / self.tau)      # B -> A pull weight
        w_ab = 1.0 - w_ba
        l_ba = ((self.pred(zs_b, wl_b, wl_a) - zs_a.detach()) ** 2).mean(1)
        l_ab = ((self.pred(zs_a, wl_a, wl_b) - zs_b.detach()) ** 2).mean(1)
        return (w_ba * l_ba + w_ab * l_ab).mean()

    def forward(self, xa, xb, qa, qb, xc=None):
        """xa, xb: (B, 4, L) observations of the same stars from two surveys.
        xc: optional third observation set (used by the *_aab variants: two good
        surveys plus the bad one, to test whether *adding* a bad survey hurts)."""
        B = xa.shape[0]
        wl_a, wl_b = xa[:, 3, 0], xb[:, 3, 0]
        v = self.variant
        if v in ("lejepa_aab", "contrastive_aab"):
            va1, va2 = two_views(xa, self.disjoint)
            vb1, vb2 = two_views(xb, self.disjoint)
            vc1, vc2 = two_views(xc, self.disjoint)
            z = self.enc(torch.cat([va1, va2, vb1, vb2, vc1, vc2]))
            zs = list(z.split(B))
            if v == "contrastive_aab":
                sid = torch.arange(B, device=z.device).repeat(6)
                loss = nt_xent_multi(z, sid, self.temperature)
                return loss, dict(align=loss.item(), reg=0.0)
            pairs = [(zs[i], zs[j]) for i in range(6) for j in range(i + 1, 6)]
            loss_align = sum(mse(p, r) for p, r in pairs) / len(pairs)
            loss_reg = sigreg(z)
            return loss_align + self.lam * loss_reg, dict(align=loss_align.item(), reg=loss_reg.item())
        if v == "aug_only":
            va1, va2 = two_views(xa, self.disjoint)
            views = torch.cat([va1, va2])
            z = self.enc(views)
            z1, z2 = z[:B], z[B:]
            loss_align = mse(z1, z2)
            loss_reg = sigreg(z)
            return loss_align + self.lam * loss_reg, dict(align=loss_align.item(), reg=loss_reg.item())

        va1, va2 = two_views(xa, self.disjoint)
        vb1, vb2 = two_views(xb, self.disjoint)
        views = torch.cat([va1, va2, vb1, vb2])
        z = self.enc(views)
        a1, a2, b1, b2 = z[:B], z[B:2 * B], z[2 * B:3 * B], z[3 * B:]

        if v in ("contrastive", "contrastive_aa"):
            sid = torch.arange(B, device=z.device).repeat(4)
            loss = nt_xent_multi(z, sid, self.temperature)
            return loss, dict(align=loss.item(), reg=0.0)

        if v in ("lejepa", "single_good", "lejepa_aa"):
            pairs = [(a1, a2), (b1, b2), (a1, b1), (a1, b2), (a2, b1), (a2, b2)]
            loss_align = sum(mse(p, r) for p, r in pairs) / len(pairs)
            loss_reg = sigreg(z)
            return loss_align + self.lam * loss_reg, dict(align=loss_align.item(), reg=loss_reg.item())

        # the split / gated family
        loss_aug = 0.5 * (mse(a1, a2) + mse(b1, b2))
        sa1, sa2, sb1, sb2 = map(self.slice, (a1, a2, b1, b2))
        cross_pairs = [(sa1, sb1), (sa1, sb2), (sa2, sb1), (sa2, sb2)]
        if v == "split_only":
            loss_cross = sum(mse(p, r) for p, r in cross_pairs) / 4
        else:  # ours, gate_only
            loss_cross = sum(self.cross_gated(p, r, qa, qb, wl_a, wl_b) for p, r in cross_pairs) / 4
        loss_reg = sigreg(z)
        loss = loss_aug + loss_cross + self.lam * loss_reg
        return loss, dict(align=loss_aug.item(), cross=loss_cross.item(), reg=loss_reg.item())


class SupervisedNet(nn.Module):
    """Ceiling / floor check: predict the latents directly from raw observations."""

    def __init__(self, n_cls: int = 5, n_reg: int = 3, width: int = 64):
        super().__init__()
        self.enc = Encoder(128, width)
        self.cls = nn.Linear(128, n_cls)
        self.reg = nn.Linear(128, n_reg)

    def forward(self, x):
        h = F.gelu(self.enc(x))
        return self.cls(h), self.reg(h)
