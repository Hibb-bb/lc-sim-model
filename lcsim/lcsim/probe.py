"""Probes on frozen embeddings, one per latent: linear (logistic / ridge) and
a small MLP. With class imbalance, accuracy is misleading, so balanced
accuracy is reported as well."""
from __future__ import annotations

import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

REG_TARGETS = ["logP", "amp_band", "fine"]
METRICS = ["cls_acc", "cls_bacc", "logP_r2", "amp_band_r2", "fine_r2"]


def probe_targets(latents: dict, obs: dict) -> dict:
    """Targets for a given observation set: class + regression latents."""
    return dict(cls=latents["cls"], logP=latents["logP"], amp_band=obs["amp_band"], fine=latents["fine"])


def _r2(pred, y):
    return float(1 - ((pred - y) ** 2).mean() / y.var())


def fit_eval(z_tr: np.ndarray, t_tr: dict, z_te: np.ndarray, t_te: dict, mlp: bool = True) -> dict:
    """Fit probes on (z_tr, t_tr), evaluate on (z_te, t_te).

    Returns {metric: score} for linear probes and, if mlp, the same metrics
    with an `_mlp` suffix for a 1-hidden-layer MLP (256 units, early stopping).
    """
    sc = StandardScaler().fit(z_tr)
    a, b = sc.transform(z_tr), sc.transform(z_te)
    out = {}
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(a, t_tr["cls"])
    p = clf.predict(b)
    out["cls_acc"] = float((p == t_te["cls"]).mean())
    out["cls_bacc"] = float(balanced_accuracy_score(t_te["cls"], p))
    for k in REG_TARGETS:
        rr = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(a, t_tr[k])
        out[f"{k}_r2"] = _r2(rr.predict(b), t_te[k])

    if mlp:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            kw = dict(hidden_layer_sizes=(256,), early_stopping=True, max_iter=300, random_state=0, alpha=1e-4)
            clf = MLPClassifier(**kw).fit(a, t_tr["cls"])
            p = clf.predict(b)
            out["cls_acc_mlp"] = float((p == t_te["cls"]).mean())
            out["cls_bacc_mlp"] = float(balanced_accuracy_score(t_te["cls"], p))
            for k in REG_TARGETS:
                mu, sd = t_tr[k].mean(), t_tr[k].std()
                reg = MLPRegressor(**kw).fit(a, (t_tr[k] - mu) / sd)
                out[f"{k}_r2_mlp"] = _r2(reg.predict(b) * sd + mu, t_te[k])
    return out
