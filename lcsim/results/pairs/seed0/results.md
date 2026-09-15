# Survey-pair experiment

Same band (550 nm), quality A > B > C: A: σ=0.015, cadence=0.5, B: σ=0.05, cadence=0.3, C: σ=0.15, cadence=0.1. 300 SSL epochs. Mean ± std over 1 seed(s). Every model is probed (on its embedding) with the same train/test stars.

## Supervised on raw observations (per-survey ceiling)

| survey | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| A | 1.000 | 1.000 | 0.874 | 0.978 | 0.872 |
| B | 0.976 | 0.975 | 0.834 | 0.962 | 0.731 |
| C | 0.500 | 0.496 | 0.360 | 0.764 | 0.133 |

## Probed on survey A test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.990 | 0.990 | 0.848 | 0.954 | 0.460 |
| lejepa | AC | 0.931 | 0.931 | 0.796 | 0.950 | 0.563 |
| lejepa_noproj | AB | 0.965 | 0.964 | 0.777 | 0.930 | 0.185 |
| lejepa_noproj | AC | 0.927 | 0.927 | 0.802 | 0.932 | 0.437 |
| contrastive | AB | 1.000 | 1.000 | 0.865 | 0.970 | 0.643 |
| contrastive | AC | 0.963 | 0.963 | 0.821 | 0.942 | 0.787 |

## Probed on survey A test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.992 | 0.992 | 0.899 | 0.986 | 0.832 |
| lejepa | AC | 0.972 | 0.972 | 0.871 | 0.979 | 0.798 |
| lejepa_noproj | AB | 0.976 | 0.975 | 0.865 | 0.968 | 0.507 |
| lejepa_noproj | AC | 0.961 | 0.961 | 0.865 | 0.972 | 0.690 |
| contrastive | AB | 0.993 | 0.993 | 0.912 | 0.990 | 0.863 |
| contrastive | AC | 0.985 | 0.984 | 0.890 | 0.984 | 0.875 |

## Probed on survey B test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.971 | 0.971 | 0.836 | 0.948 | 0.395 |
| lejepa | AC | 0.863 | 0.862 | 0.759 | 0.937 | 0.456 |
| lejepa_noproj | AB | 0.948 | 0.947 | 0.775 | 0.925 | 0.170 |
| lejepa_noproj | AC | 0.872 | 0.870 | 0.756 | 0.910 | 0.309 |
| contrastive | AB | 0.990 | 0.990 | 0.837 | 0.958 | 0.590 |
| contrastive | AC | 0.905 | 0.904 | 0.784 | 0.938 | 0.666 |

## Probed on survey B test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.978 | 0.978 | 0.895 | 0.973 | 0.687 |
| lejepa | AC | 0.920 | 0.921 | 0.857 | 0.962 | 0.594 |
| lejepa_noproj | AB | 0.963 | 0.963 | 0.854 | 0.958 | 0.404 |
| lejepa_noproj | AC | 0.895 | 0.894 | 0.831 | 0.950 | 0.407 |
| contrastive | AB | 0.996 | 0.995 | 0.887 | 0.979 | 0.739 |
| contrastive | AC | 0.944 | 0.943 | 0.871 | 0.971 | 0.735 |

## Probed on survey C test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.568 | 0.563 | 0.437 | 0.780 | 0.170 |
| lejepa | AC | 0.527 | 0.523 | 0.431 | 0.780 | 0.155 |
| lejepa_noproj | AB | 0.558 | 0.554 | 0.430 | 0.741 | 0.064 |
| lejepa_noproj | AC | 0.570 | 0.567 | 0.497 | 0.775 | 0.104 |
| contrastive | AB | 0.544 | 0.539 | 0.377 | 0.778 | 0.168 |
| contrastive | AC | 0.534 | 0.530 | 0.427 | 0.771 | 0.157 |

## Probed on survey C test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.593 | 0.589 | 0.484 | 0.781 | 0.159 |
| lejepa | AC | 0.558 | 0.556 | 0.482 | 0.788 | 0.139 |
| lejepa_noproj | AB | 0.580 | 0.577 | 0.464 | 0.746 | 0.041 |
| lejepa_noproj | AC | 0.580 | 0.577 | 0.531 | 0.784 | 0.115 |
| contrastive | AB | 0.578 | 0.574 | 0.428 | 0.778 | 0.161 |
| contrastive | AC | 0.577 | 0.574 | 0.480 | 0.779 | 0.171 |

