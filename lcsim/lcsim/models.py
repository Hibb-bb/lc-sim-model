"""Encoder, predictor, SIGReg, augmentations and the loss for every variant."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #

def projector(d_in: int, d_out: int) -> nn.Sequential:
    """LeJEPA's projector, torchvision ``MLP(d_in, [4 d_in, 4 d_in, d_out], norm_layer=BatchNorm1d)``."""
    h = 4 * d_in
    return nn.Sequential(
        nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(),
        nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU(),
        nn.Linear(h, d_out),
    )


class SplitProjector(nn.Module):
    """One projector per slice: proj(z) = [proj_s(z_s) | proj_p(z_p)].

    The first ``p_s`` projected dims depend on z_s only, so the cross-survey term
    can act there while SIGReg still sees the concatenation (and so keeps
    decorrelating the two slices)."""

    def __init__(self, d: int, d_s: int, proj_dim: int):
        super().__init__()
        self.d_s, self.p_s = d_s, proj_dim // 2
        self.s = projector(d_s, self.p_s)
        self.p = projector(d - d_s, proj_dim - self.p_s)

    def forward(self, z):
        return torch.cat([self.s(z[:, : self.d_s]), self.p(z[:, self.d_s:])], dim=1)


class Encoder(nn.Module):
    """1D CNN over the fixed grid. Input (B, 4, L): flux, mask, sigma, wavelength."""

    def __init__(self, d_out: int = 64, width: int = 64, proj_dim: int = 16, d_s: int | None = None):
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

        # d_s set: split embedding, one projector per slice (theory_notes.md section 4)
        self.proj = projector(d_out, proj_dim) if d_s is None else SplitProjector(d_out, d_s, proj_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.cat([h.mean(-1), h.amax(-1)], dim=1)
        emb = self.head(h)
        return emb, self.proj(emb)      # probe on emb, SSL loss on proj


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


class ResidualPredictor(nn.Module):
    """Unconditioned residual MLP predictor, z + MLP(z), used by lejepa_pred on the
    lower-quality side (all surveys share a band there, so no wavelength input)."""

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, z):
        return z + self.net(z)


# --------------------------------------------------------------------------- #
# SIGReg: sketched isotropic-Gaussian regularizer (Epps-Pulley on random 1-D
# projections), as in LeJEPA. Computed on the *projected* embedding.
# --------------------------------------------------------------------------- #

class SIGReg(nn.Module):
    """Epps-Pulley statistic against N(0, I) on random 1-D projections.

    Follows the LeJEPA minimal implementation: the empirical characteristic
    function is symmetric, so integrate the squared error on ``[0, t_max]`` with
    a Gaussian window and double it, instead of integrating over
    ``[-t_max, t_max]`` -- same estimate, twice the quadrature resolution.
    """

    def __init__(self, knots: int = 17, t_max: float = 3.0, n_proj: int = 256):
        super().__init__()
        self.n_proj = n_proj
        t = torch.linspace(0, t_max, knots)
        dt = t_max / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt                              # trapezoid, halved at the ends
        window = torch.exp(-t.square() / 2.0)        # cf of N(0,1); also the window
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", w * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """proj: (..., B, P). The statistic is taken over the B axis, so a
        (V, B, P) input regularizes each of the V views separately."""
        A = torch.randn(proj.size(-1), self.n_proj, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0)
        x_t = (proj @ A).unsqueeze(-1) * self.t      # (..., B, n_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


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
        lejepa       LeJEPA on the projected embedding: lam*SIGReg + (1-lam)*invariance
        contrastive  multi-positive NT-Xent over all views
        lejepa_aa / contrastive_aa    same loss, A paired with a 2nd independent A observation (good+good)
        lejepa_aab / contrastive_aab  same loss on A, A2 and B together (good+good+bad): does adding B hurt?
        aug_only     LeJEPA on augmentations of the survey-A observation only
        single_good  LeJEPA on two independent survey-A observations of each star
                     (caller passes the second A observation as xb) -- private-info ceiling
        ours         split z=[z_s|z_p], one projector per slice; view invariance on all of proj;
                     gated asymmetric predictor loss on proj_s(z_s)
        split_only   split, but symmetric cross invariance on proj_s(z_s)
        gate_only    gated asymmetric predictor loss on the full proj, no split

    Every LeJEPA-family loss follows MINIMAL.md: all alignment terms and SIGReg act on
    the projector output, and loss = lam * SIGReg + (1 - lam) * alignment, where the
    alignment term has total weight 1 (the split family averages its view and cross terms).
    """

    def __init__(self, variant: str, d: int = 64, d_s: int = 32, lam: float = 0.02,
                 tau: float = 0.5, temperature: float = 0.2, width: int = 64, disjoint: bool = True,
                 proj_dim: int = 16):
        super().__init__()
        self.disjoint = disjoint
        self.variant, self.d, self.d_s, self.lam, self.tau, self.temperature = variant, d, d_s, lam, tau, temperature
        split = variant in ("ours", "split_only")
        self.enc = Encoder(d, width, proj_dim, d_s if split else None)
        self.sigreg = SIGReg()
        self.p_s = self.enc.proj.p_s if split else proj_dim      # projected dims the cross term acts on
        self.pred = Predictor(self.p_s) if variant in ("ours", "gate_only") else None
        if variant in ("lejepa_pred", "lejepa_pred_x", "lejepa_pred_x_raw"):
            self.pred = ResidualPredictor(proj_dim)

    def lejepa(self, proj_v, inv_loss):
        """LeJEPA objective: convex combination of SIGReg and the invariance term,
        both on the projected embedding. proj_v is (V, B, P)."""
        reg = self.sigreg(proj_v)
        return self.lam * reg + (1.0 - self.lam) * inv_loss, reg

    @staticmethod
    def invariance(proj_v):
        """Spread of the V views around their per-sample mean, as in LeJEPA."""
        return (proj_v.mean(0) - proj_v).square().mean()

    def slice(self, p):
        """Shared part of the projection: proj_s(z_s) for split variants, all of it otherwise."""
        return p[..., : self.p_s]

    def cross_gated(self, ps_a, ps_b, q_a, q_b, wl_a, wl_b):
        w_ba = torch.sigmoid((q_a - q_b) / self.tau)      # B -> A pull weight
        w_ab = 1.0 - w_ba
        # 0.25 * squared error = spread of two views around their mean, i.e. LeJEPA's
        # invariance scale for a pair, so the gated term is comparable to `invariance`
        l_ba = 0.25 * ((self.pred(ps_b, wl_b, wl_a) - ps_a.detach()) ** 2).mean(1)
        l_ab = 0.25 * ((self.pred(ps_a, wl_a, wl_b) - ps_b.detach()) ** 2).mean(1)
        return (w_ba * l_ba + w_ab * l_ab).mean()

    def forward(self, xa, xb, qa, qb, xc=None):
        """xa, xb: (B, 4, L) observations of the same stars from two surveys.
        xc: optional third observation set (used by the *_aab variants: two good
        surveys plus the bad one, to test whether *adding* a bad survey hurts).

        The encoder returns (emb, proj); every SSL loss is computed on `proj`
        and every probe reads `emb`."""
        B = xa.shape[0]
        wl_a, wl_b = xa[:, 3, 0], xb[:, 3, 0]
        v = self.variant
        if v in ("lejepa_aab", "contrastive_aab"):
            va1, va2 = two_views(xa, self.disjoint)
            vb1, vb2 = two_views(xb, self.disjoint)
            vc1, vc2 = two_views(xc, self.disjoint)
            _, proj = self.enc(torch.cat([va1, va2, vb1, vb2, vc1, vc2]))
            if v == "contrastive_aab":
                sid = torch.arange(B, device=proj.device).repeat(6)
                loss = nt_xent_multi(proj, sid, self.temperature)
                return loss, dict(align=loss.item(), reg=0.0)
            proj_v = proj.reshape(6, B, -1)
            loss_align = self.invariance(proj_v)
            loss, loss_reg = self.lejepa(proj_v, loss_align)
            return loss, dict(align=loss_align.item(), reg=loss_reg.item())
        if v in ("lejepa_pred_x", "lejepa_pred_x_raw"):
            # LeJEPA + predictor on cross-survey pairs only: one view per survey per star and no
            # within-survey invariance. p(proj_b) predicts stop-grad[proj_a] (b is the lower-quality
            # side); SIGReg on both views. _x augments each single view; _x_raw does not augment.
            va, vb = (xa, xb) if v == "lejepa_pred_x_raw" else (augment(xa), augment(xb))
            _, proj = self.enc(torch.cat([va, vb]))
            proj_v = proj.reshape(2, B, -1)
            loss_cross = 0.25 * (self.pred(proj_v[1]) - proj_v[0].detach()).square().mean()
            loss, loss_reg = self.lejepa(proj_v, loss_cross)
            return loss, dict(cross=loss_cross.item(), reg=loss_reg.item())

        if v in ("contrastive_x", "contrastive_x_raw"):
            # cross-survey pairs only: one view per survey per star, and the only positive is the
            # other survey's view of the same star (never two views of the same observation).
            # _x augments each single view as usual; _x_raw feeds the observations unaugmented.
            va, vb = (xa, xb) if v == "contrastive_x_raw" else (augment(xa), augment(xb))
            _, proj = self.enc(torch.cat([va, vb]))
            sid = torch.arange(B, device=proj.device).repeat(2)
            loss = nt_xent_multi(proj, sid, self.temperature)
            return loss, dict(align=loss.item(), reg=0.0)

        if v == "aug_only":
            va1, va2 = two_views(xa, self.disjoint)
            _, proj = self.enc(torch.cat([va1, va2]))
            proj_v = proj.reshape(2, B, -1)
            loss_align = self.invariance(proj_v)
            loss, loss_reg = self.lejepa(proj_v, loss_align)
            return loss, dict(align=loss_align.item(), reg=loss_reg.item())

        va1, va2 = two_views(xa, self.disjoint)
        vb1, vb2 = two_views(xb, self.disjoint)
        views = torch.cat([va1, va2, vb1, vb2])
        emb, proj = self.enc(views)
        proj_v = proj.reshape(4, B, -1)

        if v in ("contrastive", "contrastive_aa"):
            sid = torch.arange(B, device=proj.device).repeat(4)
            loss = nt_xent_multi(proj, sid, self.temperature)
            return loss, dict(align=loss.item(), reg=0.0)

        if v == "lejepa_pred":
            # LeJEPA with a predictor on the lower-quality side (xb): within-survey view invariance
            # on proj, plus p(proj_b) predicting stop-grad[proj_a] for every cross-survey view pair,
            # so the better survey is never pulled toward the worse one. 0.25 * squared error is
            # LeJEPA's invariance scale for a pair; the two terms are averaged to total weight 1.
            loss_aug = 0.5 * (self.invariance(proj_v[:2]) + self.invariance(proj_v[2:]))
            pa1, pa2, pb1, pb2 = proj_v
            loss_cross = sum(0.25 * (self.pred(pb) - pa.detach()).square().mean()
                             for pa, pb in [(pa1, pb1), (pa1, pb2), (pa2, pb1), (pa2, pb2)]) / 4
            loss, loss_reg = self.lejepa(proj_v, 0.5 * (loss_aug + loss_cross))
            return loss, dict(align=loss_aug.item(), cross=loss_cross.item(), reg=loss_reg.item())

        if v == "lejepa_noproj":   # ablation: invariance and SIGReg on the embedding itself, no projector
            emb_v = emb.reshape(4, B, -1)
            loss_align = self.invariance(emb_v)
            loss, loss_reg = self.lejepa(emb_v, loss_align)
            return loss, dict(align=loss_align.item(), reg=loss_reg.item())

        if v in ("lejepa", "single_good", "lejepa_aa"):
            loss_align = self.invariance(proj_v)
            loss, loss_reg = self.lejepa(proj_v, loss_align)
            return loss, dict(align=loss_align.item(), reg=loss_reg.item())

        # the split / gated family, entirely on proj as in LeJEPA: within-survey view
        # invariance on every projected dim, the cross-survey term on the shared part
        # proj_s(z_s) only (all of proj for gate_only), SIGReg on the concatenation.
        # The two alignment terms are averaged so their total weight is 1, as in plain LeJEPA.
        loss_aug = 0.5 * (self.invariance(proj_v[:2]) + self.invariance(proj_v[2:]))
        sa1, sa2, sb1, sb2 = map(self.slice, proj_v)
        cross_pairs = [(sa1, sb1), (sa1, sb2), (sa2, sb1), (sa2, sb2)]
        if v == "split_only":
            loss_cross = sum(self.invariance(torch.stack([p, r])) for p, r in cross_pairs) / 4
        else:  # ours, gate_only
            loss_cross = sum(self.cross_gated(p, r, qa, qb, wl_a, wl_b) for p, r in cross_pairs) / 4
        loss, loss_reg = self.lejepa(proj_v, 0.5 * (loss_aug + loss_cross))
        return loss, dict(align=loss_aug.item(), cross=loss_cross.item(), reg=loss_reg.item())


class SupervisedNet(nn.Module):
    """Ceiling / floor check: predict the latents directly from raw observations."""

    def __init__(self, n_cls: int = 5, n_reg: int = 3, width: int = 64):
        super().__init__()
        self.enc = Encoder(128, width)
        self.cls = nn.Linear(128, n_cls)
        self.reg = nn.Linear(128, n_reg)

    def forward(self, x):
        emb, _ = self.enc(x)
        h = F.gelu(emb)
        return self.cls(h), self.reg(h)
