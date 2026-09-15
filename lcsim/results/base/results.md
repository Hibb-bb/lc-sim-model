# Results

## Supervised on raw observations (ceiling per survey)

| survey | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| A | 0.982 | 0.842 | 0.973 | 0.822 |
| B | 0.493 | 0.356 | 0.745 | 0.161 |
| C | 0.952 | 0.809 | 0.960 | 0.698 |

## Survey A (good) test observations

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.826 | 0.677 | 0.919 | 0.475 |
| contrastive | 0.893 | 0.778 | 0.934 | 0.612 |
| aug_only | 0.895 | 0.705 | 0.914 | 0.342 |
| single_good | 0.904 | 0.740 | 0.918 | 0.335 |
| split_only | 0.827 | 0.708 | 0.924 | 0.200 |
| gate_only | 0.829 | 0.697 | 0.912 | 0.252 |
| ours | 0.875 | 0.715 | 0.929 | 0.235 |

## Survey B (bad) test observations

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.491 | 0.338 | 0.737 | 0.147 |
| contrastive | 0.529 | 0.401 | 0.762 | 0.189 |
| aug_only | 0.338 | 0.119 | 0.654 | 0.112 |
| single_good | 0.394 | 0.181 | 0.669 | 0.108 |
| split_only | 0.407 | 0.239 | 0.733 | 0.041 |
| gate_only | 0.484 | 0.365 | 0.757 | 0.093 |
| ours | 0.505 | 0.358 | 0.753 | 0.101 |

## Survey C (unseen config), probes fit on C

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.713 | 0.601 | 0.896 | 0.454 |
| contrastive | 0.776 | 0.670 | 0.922 | 0.532 |
| aug_only | 0.427 | 0.285 | 0.823 | 0.377 |
| single_good | 0.412 | 0.226 | 0.715 | 0.199 |
| split_only | 0.504 | 0.305 | 0.870 | 0.179 |
| gate_only | 0.660 | 0.525 | 0.899 | 0.289 |
| ours | 0.695 | 0.586 | 0.907 | 0.282 |

## Probes fit on A, applied to C

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.531 | -3.584 | -2.857 | -2.099 |
| contrastive | 0.524 | -0.204 | -4.368 | 0.290 |
| aug_only | 0.202 | -0.363 | 0.270 | -0.138 |
| single_good | 0.232 | -0.006 | -2.116 | -0.374 |
| split_only | 0.210 | -0.893 | -0.271 | -37.749 |
| gate_only | 0.334 | -1.743 | -7.861 | -0.927 |
| ours | 0.212 | 0.172 | 0.284 | -4.054 |

## Survey A, good nights only

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.855 | 0.708 | 0.924 | 0.514 |
| contrastive | 0.913 | 0.805 | 0.944 | 0.664 |
| aug_only | 0.930 | 0.738 | 0.922 | 0.390 |
| single_good | 0.928 | 0.782 | 0.929 | 0.364 |
| split_only | 0.873 | 0.743 | 0.932 | 0.221 |
| gate_only | 0.870 | 0.733 | 0.921 | 0.284 |
| ours | 0.903 | 0.748 | 0.934 | 0.274 |

## Survey A, bad nights only

| method | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|
| lejepa | 0.659 | 0.519 | 0.890 | 0.249 |
| contrastive | 0.780 | 0.645 | 0.878 | 0.315 |
| aug_only | 0.696 | 0.540 | 0.873 | 0.066 |
| single_good | 0.767 | 0.525 | 0.864 | 0.164 |
| split_only | 0.564 | 0.529 | 0.885 | 0.077 |
| gate_only | 0.595 | 0.514 | 0.869 | 0.061 |
| ours | 0.713 | 0.550 | 0.903 | 0.009 |

## Shared vs private slice (split methods)

| method | slice | cls_acc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| split_only | A_zs | 0.672 | 0.494 | 0.713 | 0.145 |
| split_only | A_zp | 0.750 | 0.650 | 0.905 | 0.133 |
| split_only | B_zs | 0.369 | 0.155 | 0.661 | 0.016 |
| split_only | B_zp | 0.403 | 0.238 | 0.727 | 0.027 |
| ours | A_zs | 0.826 | 0.694 | 0.908 | 0.139 |
| ours | A_zp | 0.830 | 0.673 | 0.901 | 0.175 |
| ours | B_zs | 0.480 | 0.359 | 0.747 | 0.056 |
| ours | B_zp | 0.500 | 0.349 | 0.742 | 0.078 |

## Embedding health on A (std_min / std_mean / mean |corr|)

- lejepa: 0.783 / 1.135 / 0.239
- contrastive: 0.325 / 0.568 / 0.224
- aug_only: 0.529 / 1.206 / 0.227
- single_good: 0.541 / 1.201 / 0.218
- split_only: 0.108 / 1.207 / 0.191
- gate_only: 0.845 / 1.620 / 0.271
- ours: 0.649 / 1.616 / 0.271
