# Results

## Supervised on raw observations (ceiling per survey)

| survey | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| A | 0.981 | 0.840 | 0.971 | 0.808 |
| B | 0.989 | 0.847 | 0.966 | 0.775 |
| C | 0.945 | 0.796 | 0.957 | 0.686 |

## Survey A (good) test observations

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.863 | 0.694 | 0.917 | 0.447 |
| ours | 0.848 | 0.674 | 0.916 | 0.282 |

## Survey B (bad) test observations

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.860 | 0.718 | 0.917 | 0.477 |
| ours | 0.844 | 0.699 | 0.922 | 0.256 |

## Survey C (unseen config), probes fit on C

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.788 | 0.676 | 0.908 | 0.444 |
| ours | 0.764 | 0.636 | 0.914 | 0.290 |

## Probes fit on A, applied to C

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.663 | -0.131 | -2.013 | 0.090 |
| ours | 0.195 | -5.745 | -11.905 | -8.121 |

## Survey A, good nights only

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.888 | 0.738 | 0.926 | 0.494 |
| ours | 0.882 | 0.706 | 0.921 | 0.314 |

## Survey A, bad nights only

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.720 | 0.469 | 0.873 | 0.177 |
| ours | 0.652 | 0.513 | 0.889 | 0.096 |

## Shared vs private slice (split methods)

| method | slice | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| ours | A_zs | 0.820 | 0.642 | 0.905 | 0.136 |
| ours | A_zp | 0.802 | 0.619 | 0.894 | 0.100 |
| ours | B_zs | 0.818 | 0.654 | 0.900 | 0.159 |
| ours | B_zp | 0.803 | 0.660 | 0.912 | 0.141 |

## Embedding health on A (std_min / std_mean / mean |corr|)

- lejepa: 0.751 / 1.214 / 0.254
- ours: 0.584 / 1.279 / 0.316
