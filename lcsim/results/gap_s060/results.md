# Results

## Supervised on raw observations (ceiling per survey)

| survey | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| A | 0.981 | 0.840 | 0.971 | 0.808 |
| B | 0.907 | 0.768 | 0.939 | 0.576 |
| C | 0.945 | 0.796 | 0.957 | 0.686 |

## Survey A (good) test observations

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.888 | 0.683 | 0.919 | 0.450 |
| ours | 0.872 | 0.685 | 0.917 | 0.267 |

## Survey B (bad) test observations

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.798 | 0.652 | 0.897 | 0.325 |
| ours | 0.764 | 0.659 | 0.906 | 0.236 |

## Survey C (unseen config), probes fit on C

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.778 | 0.633 | 0.905 | 0.396 |
| ours | 0.740 | 0.657 | 0.920 | 0.294 |

## Probes fit on A, applied to C

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.519 | -0.930 | -2.006 | -0.763 |
| ours | 0.195 | -22.485 | -2.915 | -5.587 |

## Survey A, good nights only

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.911 | 0.706 | 0.925 | 0.498 |
| ours | 0.908 | 0.715 | 0.924 | 0.315 |

## Survey A, bad nights only

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.760 | 0.565 | 0.886 | 0.176 |
| ours | 0.662 | 0.532 | 0.876 | -0.011 |

## Shared vs private slice (split methods)

| method | slice | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| ours | A_zs | 0.838 | 0.641 | 0.900 | 0.184 |
| ours | A_zp | 0.821 | 0.635 | 0.900 | 0.161 |
| ours | B_zs | 0.731 | 0.643 | 0.892 | 0.143 |
| ours | B_zp | 0.731 | 0.627 | 0.889 | 0.162 |

## Embedding health on A (std_min / std_mean / mean |corr|)

- lejepa: 0.731 / 1.210 / 0.255
- ours: 0.742 / 1.402 / 0.316
