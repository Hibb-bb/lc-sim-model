#!/usr/bin/env python3
"""Cross-survey representation learning on synthetic variable stars.

Trains several SSL variants on paired observations (survey A = good, survey B
= bad) of the same stars, then probes each latent linearly on embeddings of
A, B and an unseen survey C.

    python run_experiment.py --out results/base
    python run_experiment.py --sigma_bad 0.03 --out results/gap_small
"""
from __future__ import annotations

import argparse, functools, json, os, time
import numpy as np

from lcsim.simulate import SimConfig, SurveyConfig, make_dataset, observe
from lcsim.train import train_ssl, embed, train_supervised, supervised_eval
from lcsim.probe import probe_targets, fit_eval, METRICS

VARIANTS = ["lejepa", "contrastive", "aug_only", "single_good", "split_only", "gate_only", "ours"]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n_train", type=int, default=8000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--d_s", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--lam", type=float, default=50.0, help="SIGReg weight")
    p.add_argument("--tau", type=float, default=0.5, help="quality-gate temperature")
    p.add_argument("--variants", type=str, default=",".join(VARIANTS))
    p.add_argument("--no_supervised", action="store_true")
    p.add_argument("--overlap_views", action="store_true", help="use overlapping crops instead of disjoint-epoch views")
    # survey A (good), B (bad), C (unseen)
    p.add_argument("--sigma_good", type=float, default=0.015)
    p.add_argument("--keep_good", type=float, default=0.5)
    p.add_argument("--bad_prob_good", type=float, default=0.15, help="fraction of bad nights in survey A")
    p.add_argument("--sigma_bad", type=float, default=0.15)
    p.add_argument("--keep_bad", type=float, default=0.10)
    p.add_argument("--smooth_bad", type=float, default=0.0, help="boxcar integration time (days) of survey B")
    p.add_argument("--wl_good", type=float, default=477.0)
    p.add_argument("--wl_bad", type=float, default=623.0)
    p.add_argument("--fine_max", type=float, default=0.6, help="max bump amplitude relative to amp")
    p.add_argument("--bump_width", type=float, default=0.06)
    p.add_argument("--class_probs", type=str, default="", help='comma-separated class priors, e.g. "0.05,0.1,0.75,0.05,0.05"')
    p.add_argument("--no_mlp", action="store_true", help="skip the MLP probes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="results/base")
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "w")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    cp = tuple(float(v) for v in args.class_probs.split(",")) if args.class_probs else None
    cfg = SimConfig(seed=args.seed, fine_range=(0.0, args.fine_max), bump_width=args.bump_width, class_probs=cp)
    A = SurveyConfig("A", args.wl_good, args.sigma_good, args.keep_good, bad_prob=args.bad_prob_good)
    B = SurveyConfig("B", args.wl_bad, args.sigma_bad, args.keep_bad, smooth_days=args.smooth_bad)
    C = SurveyConfig("C", 800.0, 0.05, 0.3)                    # never seen in training
    A2 = SurveyConfig("A2", args.wl_good, args.sigma_good, args.keep_good, bad_prob=args.bad_prob_good)  # 2nd independent A obs
    log(f"surveys: {A}\n         {B}\n         {C}")

    t0 = time.time()
    tr = make_dataset(args.n_train, [A, B, C, A2], cfg, seed=args.seed + 1)
    te = make_dataset(args.n_test, [A, B, C], cfg, seed=args.seed + 2)
    log(f"data: {args.n_train} train / {args.n_test} test stars, {time.time() - t0:.1f}s")
    log(f"quality score q: A={tr['obs']['A']['q'].mean():.2f}±{tr['obs']['A']['q'].std():.2f} "
        f"(bad nights {tr['obs']['A']['q'][tr['obs']['A']['is_bad']].mean():.2f}), "
        f"B={tr['obs']['B']['q'].mean():.2f}, C={tr['obs']['C']['q'].mean():.2f}")

    results = {"args": vars(args), "probes": {}, "supervised": {}}
    log(f"class counts (train): {np.bincount(tr['latents']['cls'], minlength=5).tolist()}")
    fe = functools.partial(fit_eval, mlp=not args.no_mlp)
    tgt = {s: (probe_targets(tr["latents"], tr["obs"][s]), probe_targets(te["latents"], te["obs"][s]))
           for s in "ABC"}

    # ---------------- supervised ceilings on raw observations ----------------
    if not args.no_supervised:
        for s in "ABC":
            t_tr, t_te = tgt[s]
            reg_tr = np.stack([t_tr[k] for k in ("logP", "amp_band", "fine")], 1).astype(np.float32)
            reg_te = np.stack([t_te[k] for k in ("logP", "amp_band", "fine")], 1).astype(np.float32)
            mu, sd = reg_tr.mean(0), reg_tr.std(0)
            net = train_supervised(tr["obs"][s]["x"], t_tr["cls"], (reg_tr - mu) / sd,
                                   epochs=args.epochs, batch=args.batch, seed=args.seed, log=log, width=args.width)
            acc, bacc, r2 = supervised_eval(net, te["obs"][s]["x"], t_te["cls"], (reg_te - mu) / sd)
            results["supervised"][s] = dict(cls_acc=float(acc), cls_bacc=float(bacc), logP_r2=float(r2[0]),
                                            amp_band_r2=float(r2[1]), fine_r2=float(r2[2]))
            log(f"supervised raw {s}: " + " ".join(f"{k}={v:.3f}" for k, v in results["supervised"][s].items()))

    # ---------------- SSL variants ----------------
    xa, xb = tr["obs"]["A"]["x"], tr["obs"]["B"]["x"]
    qa, qb = tr["obs"]["A"]["q"], tr["obs"]["B"]["q"]
    for v in [v for v in args.variants.split(",") if v]:
        log(f"\n=== {v} ===")
        xc_v = None
        if v in ("single_good", "lejepa_aa", "contrastive_aa"):   # good+good: A paired with an independent A observation
            xb_v, qb_v = tr["obs"]["A2"]["x"], tr["obs"]["A2"]["q"]
        elif v in ("lejepa_aab", "contrastive_aab"):              # good+good+bad
            xb_v, qb_v, xc_v = tr["obs"]["A2"]["x"], tr["obs"]["A2"]["q"], xb
        else:
            xb_v, qb_v = xb, qb
        model = train_ssl(v, xa, xb_v, qa, qb_v, xc=xc_v, epochs=args.epochs, batch=args.batch, seed=args.seed, log=log,
                          d=args.d, d_s=args.d_s, lam=args.lam, tau=args.tau, width=args.width, disjoint=not args.overlap_views)
        z_tr = {s: embed(model, tr["obs"][s]["x"]) for s in "ABC"}
        z_te = {s: embed(model, te["obs"][s]["x"]) for s in "ABC"}
        res = {}
        for s in "ABC":
            res[s] = fe(z_tr[s], tgt[s][0], z_te[s], tgt[s][1])
        # A-fit probes applied to C: does the linear readout transfer to the unseen config?
        res["A->C"] = fe(z_tr["A"], tgt["A"][0], z_te["C"], tgt["C"][1])
        # good nights vs bad nights within survey A
        bad = te["obs"]["A"]["is_bad"]
        sub = lambda t, m: {k: val[m] for k, val in t.items()}
        res["A_goodnights"] = fe(z_tr["A"], tgt["A"][0], z_te["A"][~bad], sub(tgt["A"][1], ~bad))
        res["A_badnights"] = fe(z_tr["A"], tgt["A"][0], z_te["A"][bad], sub(tgt["A"][1], bad))
        if v in ("ours", "split_only"):
            ds = args.d_s
            res["A_zs"] = fe(z_tr["A"][:, :ds], tgt["A"][0], z_te["A"][:, :ds], tgt["A"][1])
            res["A_zp"] = fe(z_tr["A"][:, ds:], tgt["A"][0], z_te["A"][:, ds:], tgt["A"][1])
            res["B_zs"] = fe(z_tr["B"][:, :ds], tgt["B"][0], z_te["B"][:, :ds], tgt["B"][1])
            res["B_zp"] = fe(z_tr["B"][:, ds:], tgt["B"][0], z_te["B"][:, ds:], tgt["B"][1])
        # embedding health: per-dim std and mean |corr|
        zc = z_te["A"] - z_te["A"].mean(0)
        corr = np.corrcoef(zc.T); np.fill_diagonal(corr, 0)
        res["health"] = dict(std_min=float(z_te["A"].std(0).min()), std_mean=float(z_te["A"].std(0).mean()),
                             mean_abs_corr=float(np.abs(corr).mean()))
        results["probes"][v] = res
        for k, r in res.items():
            log(f"  {k:13s} " + " ".join(f"{kk}={vv:.3f}" for kk, vv in r.items()))
        json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)

    write_table(results, os.path.join(args.out, "results.md"))
    log(f"\ndone in {time.time() - t0:.0f}s -> {args.out}/results.md")


def write_table(results, path):
    metrics = METRICS
    lines = ["# Results", ""]
    if results["supervised"]:
        lines += ["## Supervised on raw observations (ceiling per survey)", "",
                  "| survey | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
        for s, r in results["supervised"].items():
            lines.append(f"| {s} | " + " | ".join(f"{r.get(m, float('nan')):.3f}" for m in metrics) + " |")
        lines.append("")
    for block, title in [("A", "Survey A (good) test observations"), ("B", "Survey B (bad) test observations"),
                         ("C", "Survey C (unseen config), probes fit on C"), ("A->C", "Probes fit on A, applied to C"),
                         ("A_goodnights", "Survey A, good nights only"), ("A_badnights", "Survey A, bad nights only")]:
        lines += [f"## {title}", "", "| method | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
        for v, res in results["probes"].items():
            r = res[block]
            lines.append(f"| {v} | " + " | ".join(f"{r[m]:.3f}" for m in metrics) + " |")
        lines.append("")
    any_v = next(iter(results["probes"].values()), {})
    if any("_mlp" in k for k in any_v.get("A", {})):
        mm = [m + "_mlp" for m in metrics]
        for block, title in [("A", "Survey A (good), MLP probes"), ("B", "Survey B (bad), MLP probes"), ("C", "Survey C (unseen), MLP probes")]:
            lines += [f"## {title}", "", "| method | " + " | ".join(mm) + " |", "|---|" + "---|" * len(mm)]
            for v, res in results["probes"].items():
                lines.append(f"| {v} | " + " | ".join(f"{res[block][m]:.3f}" for m in mm) + " |")
            lines.append("")
    lines += ["## Shared vs private slice (split methods)", "", "| method | slice | " + " | ".join(metrics) + " |",
              "|---|---|" + "---|" * len(metrics)]
    for v, res in results["probes"].items():
        for k in ("A_zs", "A_zp", "B_zs", "B_zp"):
            if k in res:
                lines.append(f"| {v} | {k} | " + " | ".join(f"{res[k][m]:.3f}" for m in metrics) + " |")
    lines += ["", "## Embedding health on A (std_min / std_mean / mean |corr|)", ""]
    for v, res in results["probes"].items():
        h = res["health"]
        lines.append(f"- {v}: {h['std_min']:.3f} / {h['std_mean']:.3f} / {h['mean_abs_corr']:.3f}")
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
