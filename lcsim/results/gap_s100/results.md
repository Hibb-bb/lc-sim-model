# Results

## Supervised on raw observations (ceiling per survey)

| survey | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| A | 0.981 | 0.980 | 0.840 | 0.971 | 0.808 |
| B | 0.735 | 0.731 | 0.623 | 0.875 | 0.339 |
| C | 0.945 | 0.944 | 0.796 | 0.957 | 0.686 |

## Survey A (good) test observations

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa | 0.849 | 0.848 | 0.674 | 0.914 | 0.458 |
| ours | 0.844 | 0.842 | 0.726 | 0.926 | 0.202 |

## Survey B (bad) test observations

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa | 0.670 | 0.667 | 0.530 | 0.850 | 0.282 |
| ours | 0.635 | 0.631 | 0.555 | 0.866 | 0.189 |

## Survey C (unseen config), probes fit on C

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa | 0.731 | 0.727 | 0.641 | 0.908 | 0.431 |
| ours | 0.694 | 0.691 | 0.592 | 0.906 | 0.375 |

## Probes fit on A, applied to C

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa | 0.549 | 0.543 | 0.340 | -3.621 | -0.746 |
| ours | 0.184 | 0.190 | -2.179 | 0.091 | -10.199 |

## Survey A, good nights only

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa | 0.877 | 0.877 | 0.706 | 0.923 | 0.503 |
| ours | 0.880 | 0.880 | 0.762 | 0.934 | 0.240 |

## Survey A, bad nights only

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa | 0.689 | 0.681 | 0.513 | 0.863 | 0.200 |
| ours | 0.635 | 0.627 | 0.543 | 0.887 | -0.023 |

## Survey A (good), MLP probes

| method | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|
| lejepa | 0.900 | 0.899 | 0.791 | 0.952 | 0.590 |
| ours | 0.862 | 0.861 | 0.801 | 0.952 | 0.415 |

## Survey B (bad), MLP probes

| method | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|
| lejepa | 0.668 | 0.665 | 0.614 | 0.864 | 0.288 |
| ours | 0.630 | 0.626 | 0.609 | 0.869 | 0.214 |

## Survey C (unseen), MLP probes

| method | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|
| lejepa | 0.792 | 0.790 | 0.755 | 0.926 | 0.464 |
| ours | 0.754 | 0.752 | 0.701 | 0.929 | 0.424 |

## Shared vs private slice (split methods)

| method | slice | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|
| ours | A_zs | 0.797 | 0.795 | 0.685 | 0.893 | 0.120 |
| ours | A_zp | 0.803 | 0.802 | 0.689 | 0.895 | 0.127 |
| ours | B_zs | 0.611 | 0.605 | 0.532 | 0.855 | 0.116 |
| ours | B_zp | 0.608 | 0.603 | 0.532 | 0.861 | 0.127 |

## Embedding health on A (std_min / std_mean / mean |corr|)

- lejepa: 0.709 / 1.173 / 0.258
- ours: 0.728 / 1.548 / 0.298
