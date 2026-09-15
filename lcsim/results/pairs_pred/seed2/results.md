# Survey-pair experiment

Same band (550 nm), quality A > B > C: A: σ=0.015, cadence=0.5, B: σ=0.05, cadence=0.3, C: σ=0.15, cadence=0.1. 300 SSL epochs. Mean ± std over 1 seed(s). Every model is probed (on its embedding) with the same train/test stars.

## Probed on survey A test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.998 | 0.998 | 0.850 | 0.952 | 0.458 |
| lejepa_pred | AC | 0.973 | 0.973 | 0.826 | 0.945 | 0.587 |

## Probed on survey A test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 1.000 | 1.000 | 0.915 | 0.986 | 0.796 |
| lejepa_pred | AC | 0.993 | 0.993 | 0.899 | 0.985 | 0.816 |

## Probed on survey B test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.996 | 0.996 | 0.846 | 0.948 | 0.459 |
| lejepa_pred | AC | 0.933 | 0.933 | 0.764 | 0.929 | 0.487 |

## Probed on survey B test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.995 | 0.995 | 0.902 | 0.972 | 0.680 |
| lejepa_pred | AC | 0.952 | 0.952 | 0.864 | 0.966 | 0.634 |

## Probed on survey C test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.546 | 0.545 | 0.295 | 0.788 | 0.158 |
| lejepa_pred | AC | 0.557 | 0.556 | 0.405 | 0.788 | 0.162 |

## Probed on survey C test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.548 | 0.547 | 0.334 | 0.789 | 0.152 |
| lejepa_pred | AC | 0.592 | 0.592 | 0.483 | 0.794 | 0.142 |

