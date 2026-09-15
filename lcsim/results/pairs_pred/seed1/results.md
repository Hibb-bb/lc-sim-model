# Survey-pair experiment

Same band (550 nm), quality A > B > C: A: σ=0.015, cadence=0.5, B: σ=0.05, cadence=0.3, C: σ=0.15, cadence=0.1. 300 SSL epochs. Mean ± std over 1 seed(s). Every model is probed (on its embedding) with the same train/test stars.

## Probed on survey A test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.999 | 0.999 | 0.860 | 0.942 | 0.510 |
| lejepa_pred | AC | 0.990 | 0.990 | 0.840 | 0.946 | 0.487 |

## Probed on survey A test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.999 | 0.999 | 0.922 | 0.987 | 0.843 |
| lejepa_pred | AC | 0.993 | 0.993 | 0.907 | 0.984 | 0.807 |

## Probed on survey B test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.990 | 0.990 | 0.840 | 0.948 | 0.445 |
| lejepa_pred | AC | 0.943 | 0.944 | 0.809 | 0.940 | 0.451 |

## Probed on survey B test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.992 | 0.992 | 0.900 | 0.974 | 0.681 |
| lejepa_pred | AC | 0.963 | 0.963 | 0.887 | 0.970 | 0.642 |

## Probed on survey C test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.546 | 0.552 | 0.383 | 0.769 | 0.143 |
| lejepa_pred | AC | 0.568 | 0.573 | 0.445 | 0.789 | 0.146 |

## Probed on survey C test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa_pred | AB | 0.564 | 0.570 | 0.392 | 0.764 | 0.132 |
| lejepa_pred | AC | 0.584 | 0.589 | 0.502 | 0.793 | 0.156 |

