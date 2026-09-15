"""Training loops for the SSL variants and the supervised ceilings, plus
embedding extraction. Everything runs on `device` (CUDA when available); the
full dataset is moved to the device once, since it is small."""
from __future__ import annotations

import math
import time
import numpy as np
import torch
import torch.nn.functional as F

from .models import Method, SupervisedNet, augment

torch.set_num_threads(max(1, torch.get_num_threads()))


def get_device(device: str | None = None) -> torch.device:
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _cosine(step, total, base_lr, warmup=50):
    if step < warmup:
        return base_lr * step / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


def train_ssl(variant: str, xa: np.ndarray, xb: np.ndarray, qa: np.ndarray, qb: np.ndarray,
              epochs: int = 20, batch: int = 256, lr: float = 2e-3, wd: float = 5e-2, seed: int = 0,
              log=print, xc: np.ndarray | None = None, device: str | None = None, **method_kw) -> Method:
    """Defaults follow LeJEPA's MINIMAL.md: AdamW(lr=2e-3, wd=5e-2), batch 256,
    one epoch of linear warmup then cosine decay."""
    dev = get_device(device)
    torch.manual_seed(seed)
    model = Method(variant, **method_kw).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(xa)
    steps_per_epoch = n // batch
    total = epochs * steps_per_epoch
    xa_t, xb_t = torch.from_numpy(xa).to(dev), torch.from_numpy(xb).to(dev)
    qa_t, qb_t = torch.from_numpy(qa).to(dev), torch.from_numpy(qb).to(dev)
    xc_t = torch.from_numpy(xc).to(dev) if xc is not None else None
    step, t0 = 0, time.time()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        agg = {}
        for i in range(steps_per_epoch):
            idx = perm[i * batch:(i + 1) * batch]
            for g in opt.param_groups:
                g["lr"] = _cosine(step, total, lr, warmup=steps_per_epoch)
            loss, parts = model(xa_t[idx], xb_t[idx], qa_t[idx], qb_t[idx],
                                xc=None if xc_t is None else xc_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / steps_per_epoch
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            log(f"  [{variant}] epoch {ep + 1}/{epochs} " + " ".join(f"{k}={v:.3f}" for k, v in agg.items())
                + f"  ({time.time() - t0:.0f}s)")
    model.eval()
    return model


@torch.no_grad()
def embed(model: Method, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(x), batch):
        out.append(model.enc(torch.from_numpy(x[i:i + batch]).to(dev))[0].cpu().numpy())   # emb, not proj
    return np.concatenate(out)


def train_supervised(x: np.ndarray, cls: np.ndarray, reg: np.ndarray, epochs: int = 20,
                     batch: int = 256, lr: float = 1e-3, seed: int = 0, log=print, width: int = 48,
                     device: str | None = None) -> SupervisedNet:
    """reg: (n, 3) standardized regression targets."""
    dev = get_device(device)
    torch.manual_seed(seed)
    model = SupervisedNet(n_reg=reg.shape[1], width=width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(x)
    steps_per_epoch = n // batch
    total = epochs * steps_per_epoch
    x_t = torch.from_numpy(x).to(dev)
    c_t, r_t = torch.from_numpy(cls).long().to(dev), torch.from_numpy(reg).float().to(dev)
    step = 0
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(steps_per_epoch):
            idx = perm[i * batch:(i + 1) * batch]
            for g in opt.param_groups:
                g["lr"] = _cosine(step, total, lr)
            lc, lr_ = model(augment(x_t[idx], crop_min=0.8, drop_max=0.1, noise_max=0.3))
            loss = F.cross_entropy(lc, c_t[idx]) + F.mse_loss(lr_, r_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
    model.eval()
    return model


@torch.no_grad()
def supervised_eval(model: SupervisedNet, x: np.ndarray, cls: np.ndarray, reg: np.ndarray, batch: int = 4096):
    model.eval()
    dev = next(model.parameters()).device
    lc, lr_ = [], []
    for i in range(0, len(x), batch):
        a, b = model(torch.from_numpy(x[i:i + batch]).to(dev))
        lc.append(a.cpu().numpy()); lr_.append(b.cpu().numpy())
    lc, lr_ = np.concatenate(lc), np.concatenate(lr_)
    pred = lc.argmax(1)
    acc = (pred == cls).mean()
    from sklearn.metrics import balanced_accuracy_score
    bacc = balanced_accuracy_score(cls, pred)
    r2 = 1 - ((lr_ - reg) ** 2).mean(0) / reg.var(0)
    return acc, bacc, r2
