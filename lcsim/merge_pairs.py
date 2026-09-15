#!/usr/bin/env python3
"""Pool per-seed run_pairs.py outputs (possibly from different --variants runs) into one
results.json / results.md. Probes are merged per seed, so runs of new variants on the
same seeds extend the existing table instead of replacing it.

    python merge_pairs.py results/pairs_unseen results/pairs_unseen/seed0 results/pairs_unseen/seed1 results/pairs_unseen/seed2
"""
import json, os, sys

from run_pairs import write_summary

out, dirs = sys.argv[1], sys.argv[2:]
merged, variants, pairs = None, [], []
for d in dirs:
    r = json.load(open(os.path.join(d, "results.json")))
    if merged is None:
        merged = {"args": dict(r["args"]), "runs": {}}
    for v in r["args"]["variants"].split(","):
        if v and v not in variants:
            variants.append(v)
    for p in r["args"]["pairs"].split(","):
        if p and p not in pairs:
            pairs.append(p)
    for seed, run in r["runs"].items():
        m = merged["runs"].setdefault(seed, {"supervised": {}, "probes": {}})
        m["supervised"] = m["supervised"] or run["supervised"]
        m["probes"].update(run["probes"])
merged["args"].update(variants=",".join(variants), pairs=",".join(pairs), out=out,
                      seeds=",".join(sorted(merged["runs"])))
os.makedirs(out, exist_ok=True)
json.dump(merged, open(os.path.join(out, "results.json"), "w"), indent=1)
write_summary(merged, os.path.join(out, "results.md"))
print("merged seeds", merged["args"]["seeds"], "variants", merged["args"]["variants"], "->", os.path.join(out, "results.md"))
