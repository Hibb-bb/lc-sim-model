# Survey-pair experiment

Same band (550 nm), quality A > B > C: A: σ=0.015, cadence=0.5, B: σ=0.05, cadence=0.3, C: σ=0.15, cadence=0.1. 300 SSL epochs. Mean ± std over 1 seed(s). Every model is probed (on its embedding) with the same train/test stars.

## Probed on survey A test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.998 | 0.997 | 0.856 | 0.955 | 0.499 |
| lejepa_pred | AC | 0.981 | 0.980 | 0.841 | 0.935 | 0.564 |

## Probed on survey A test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.995 | 0.995 | 0.907 | 0.986 | 0.842 |
| lejepa_pred | AC | 0.990 | 0.990 | 0.894 | 0.984 | 0.824 |

## Probed on survey B test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.993 | 0.992 | 0.848 | 0.946 | 0.475 |
| lejepa_pred | AC | 0.928 | 0.928 | 0.795 | 0.925 | 0.470 |

## Probed on survey B test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.990 | 0.989 | 0.897 | 0.973 | 0.681 |
| lejepa_pred | AC | 0.959 | 0.959 | 0.872 | 0.966 | 0.624 |

## Probed on survey C test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.562 | 0.558 | 0.397 | 0.784 | 0.173 |
| lejepa_pred | AC | 0.547 | 0.543 | 0.433 | 0.779 | 0.154 |

## Probed on survey C test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.603 | 0.600 | 0.433 | 0.781 | 0.159 |
| lejepa_pred | AC | 0.588 | 0.585 | 0.512 | 0.790 | 0.134 |

