# Survey-pair experiment

Same band (550 nm), quality A > B > C: A: σ=0.015, cadence=0.5, B: σ=0.05, cadence=0.3, C: σ=0.15, cadence=0.1. 300 SSL epochs. Mean ± std over 1 seed(s). Every model is probed (on its embedding) with the same train/test stars.

## Supervised on raw observations (per-survey ceiling)

| survey | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| A | 1.000 | 0.999 | 0.889 | 0.978 | 0.875 |
| B | 0.967 | 0.967 | 0.851 | 0.961 | 0.725 |
| C | 0.486 | 0.492 | 0.376 | 0.776 | 0.120 |

## Probed on survey A test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.992 | 0.992 | 0.865 | 0.954 | 0.568 |
| lejepa | AC | 0.935 | 0.936 | 0.804 | 0.944 | 0.590 |
| lejepa_noproj | AB | 0.954 | 0.954 | 0.812 | 0.909 | 0.246 |
| lejepa_noproj | AC | 0.909 | 0.910 | 0.803 | 0.935 | 0.451 |
| contrastive | AB | 1.000 | 0.999 | 0.871 | 0.967 | 0.655 |
| contrastive | AC | 0.960 | 0.960 | 0.827 | 0.937 | 0.765 |

## Probed on survey A test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.996 | 0.996 | 0.917 | 0.985 | 0.832 |
| lejepa | AC | 0.975 | 0.975 | 0.893 | 0.978 | 0.801 |
| lejepa_noproj | AB | 0.981 | 0.981 | 0.898 | 0.973 | 0.548 |
| lejepa_noproj | AC | 0.965 | 0.965 | 0.879 | 0.972 | 0.674 |
| contrastive | AB | 0.997 | 0.997 | 0.925 | 0.988 | 0.871 |
| contrastive | AC | 0.989 | 0.989 | 0.910 | 0.985 | 0.882 |

## Probed on survey B test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.981 | 0.981 | 0.840 | 0.944 | 0.497 |
| lejepa | AC | 0.859 | 0.861 | 0.772 | 0.937 | 0.460 |
| lejepa_noproj | AB | 0.937 | 0.938 | 0.796 | 0.905 | 0.234 |
| lejepa_noproj | AC | 0.837 | 0.839 | 0.755 | 0.906 | 0.315 |
| contrastive | AB | 0.990 | 0.990 | 0.847 | 0.958 | 0.615 |
| contrastive | AC | 0.911 | 0.913 | 0.792 | 0.938 | 0.652 |

## Probed on survey B test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.986 | 0.986 | 0.905 | 0.974 | 0.686 |
| lejepa | AC | 0.914 | 0.916 | 0.864 | 0.961 | 0.590 |
| lejepa_noproj | AB | 0.956 | 0.957 | 0.873 | 0.955 | 0.460 |
| lejepa_noproj | AC | 0.899 | 0.901 | 0.836 | 0.941 | 0.437 |
| contrastive | AB | 0.994 | 0.994 | 0.906 | 0.976 | 0.757 |
| contrastive | AC | 0.970 | 0.970 | 0.887 | 0.970 | 0.719 |

## Probed on survey C test observations (linear probes)

| method | trained on | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.583 | 0.588 | 0.392 | 0.776 | 0.151 |
| lejepa | AC | 0.559 | 0.565 | 0.443 | 0.782 | 0.145 |
| lejepa_noproj | AB | 0.568 | 0.571 | 0.439 | 0.756 | 0.079 |
| lejepa_noproj | AC | 0.575 | 0.581 | 0.496 | 0.794 | 0.128 |
| contrastive | AB | 0.543 | 0.548 | 0.387 | 0.773 | 0.178 |
| contrastive | AC | 0.526 | 0.533 | 0.443 | 0.778 | 0.161 |

## Probed on survey C test observations (MLP probes)

| method | trained on | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|---|
| lejepa | AB | 0.579 | 0.584 | 0.401 | 0.774 | 0.140 |
| lejepa | AC | 0.569 | 0.577 | 0.497 | 0.791 | 0.139 |
| lejepa_noproj | AB | 0.579 | 0.583 | 0.453 | 0.754 | 0.045 |
| lejepa_noproj | AC | 0.573 | 0.577 | 0.523 | 0.793 | 0.108 |
| contrastive | AB | 0.591 | 0.595 | 0.405 | 0.769 | 0.174 |
| contrastive | AC | 0.587 | 0.593 | 0.475 | 0.788 | 0.157 |

