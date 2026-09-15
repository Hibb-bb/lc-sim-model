# Survey-pair experiment

Same band (550 nm), quality A > B > C: A: σ=0.015, cadence=0.5, B: σ=0.05, cadence=0.3, C: σ=0.15, cadence=0.1. 300 SSL epochs. Mean ± std over 1 seed(s). Every model is probed (on its embedding) with the same train/test stars.

## Supervised on raw observations (per-survey ceiling)

| survey | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| A | 1.000 | 1.000 | 0.880 | 0.980 | 0.862 |
| B | 0.965 | 0.965 | 0.841 | 0.963 | 0.726 |
| C | 0.495 | 0.494 | 0.354 | 0.789 | 0.164 |

## Probed on survey A test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.996 | 0.995 | 0.830 | 0.956 | 0.467 |
| lejepa | AC | 0.948 | 0.949 | 0.799 | 0.947 | 0.559 |
| lejepa_noproj | AB | 0.958 | 0.959 | 0.774 | 0.924 | 0.272 |
| lejepa_noproj | AC | 0.909 | 0.910 | 0.791 | 0.937 | 0.447 |
| contrastive | AB | 1.000 | 1.000 | 0.866 | 0.966 | 0.638 |
| contrastive | AC | 0.959 | 0.959 | 0.838 | 0.941 | 0.763 |

## Probed on survey A test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.997 | 0.997 | 0.904 | 0.988 | 0.807 |
| lejepa | AC | 0.972 | 0.973 | 0.882 | 0.984 | 0.791 |
| lejepa_noproj | AB | 0.978 | 0.978 | 0.863 | 0.968 | 0.552 |
| lejepa_noproj | AC | 0.968 | 0.968 | 0.870 | 0.975 | 0.679 |
| contrastive | AB | 0.999 | 0.998 | 0.911 | 0.990 | 0.845 |
| contrastive | AC | 0.981 | 0.981 | 0.897 | 0.986 | 0.879 |

## Probed on survey B test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.985 | 0.985 | 0.824 | 0.949 | 0.421 |
| lejepa | AC | 0.875 | 0.876 | 0.772 | 0.932 | 0.475 |
| lejepa_noproj | AB | 0.934 | 0.935 | 0.763 | 0.922 | 0.252 |
| lejepa_noproj | AC | 0.851 | 0.852 | 0.744 | 0.914 | 0.325 |
| contrastive | AB | 0.992 | 0.992 | 0.849 | 0.960 | 0.589 |
| contrastive | AC | 0.918 | 0.918 | 0.802 | 0.932 | 0.644 |

## Probed on survey B test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.990 | 0.990 | 0.896 | 0.972 | 0.668 |
| lejepa | AC | 0.923 | 0.924 | 0.856 | 0.962 | 0.623 |
| lejepa_noproj | AB | 0.942 | 0.942 | 0.845 | 0.953 | 0.484 |
| lejepa_noproj | AC | 0.900 | 0.902 | 0.835 | 0.949 | 0.418 |
| contrastive | AB | 0.995 | 0.995 | 0.901 | 0.979 | 0.727 |
| contrastive | AC | 0.954 | 0.954 | 0.880 | 0.968 | 0.720 |

## Probed on survey C test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.549 | 0.548 | 0.374 | 0.792 | 0.146 |
| lejepa | AC | 0.541 | 0.541 | 0.402 | 0.799 | 0.183 |
| lejepa_noproj | AB | 0.548 | 0.548 | 0.365 | 0.751 | 0.060 |
| lejepa_noproj | AC | 0.576 | 0.576 | 0.462 | 0.791 | 0.109 |
| contrastive | AB | 0.570 | 0.569 | 0.379 | 0.798 | 0.176 |
| contrastive | AC | 0.541 | 0.541 | 0.368 | 0.778 | 0.190 |

## Probed on survey C test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.583 | 0.581 | 0.414 | 0.791 | 0.110 |
| lejepa | AC | 0.560 | 0.560 | 0.465 | 0.805 | 0.182 |
| lejepa_noproj | AB | 0.573 | 0.573 | 0.395 | 0.760 | 0.041 |
| lejepa_noproj | AC | 0.586 | 0.586 | 0.478 | 0.797 | 0.112 |
| contrastive | AB | 0.610 | 0.608 | 0.421 | 0.797 | 0.168 |
| contrastive | AC | 0.572 | 0.572 | 0.465 | 0.798 | 0.179 |

