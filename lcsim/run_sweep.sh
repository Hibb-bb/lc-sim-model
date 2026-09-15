#!/bin/sh
# quality-gap sweep: bad-survey noise / cadence, key variants only (single_good comes from results/base)
while kill -0 905 2>/dev/null; do sleep 10; done
python3 run_experiment.py --epochs 20 --sigma_bad 0.03 --keep_bad 0.30 --variants lejepa,ours --out results/gap_s030 > results_gap_s030.log 2>&1
python3 run_experiment.py --epochs 20 --sigma_bad 0.06 --keep_bad 0.20 --variants lejepa,ours --out results/gap_s060 > results_gap_s060.log 2>&1
python3 run_experiment.py --epochs 20 --sigma_bad 0.10 --keep_bad 0.15 --variants lejepa,ours --out results/gap_s100 > results_gap_s100.log 2>&1
