# Results

## Survey A (good) test observations

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa_aa | 0.904 | 0.904 | 0.740 | 0.918 | 0.335 |
| lejepa_aab | 0.876 | 0.875 | 0.742 | 0.931 | 0.560 |
| contrastive_aa | 0.969 | 0.969 | 0.784 | 0.913 | 0.461 |
| contrastive_aab | 0.936 | 0.935 | 0.797 | 0.931 | 0.622 |

## Survey B (bad) test observations

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa_aa | 0.394 | 0.390 | 0.181 | 0.669 | 0.108 |
| lejepa_aab | 0.494 | 0.489 | 0.360 | 0.731 | 0.161 |
| contrastive_aa | 0.421 | 0.418 | 0.202 | 0.696 | 0.145 |
| contrastive_aab | 0.546 | 0.542 | 0.409 | 0.755 | 0.194 |

## Survey C (unseen config), probes fit on C

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa_aa | 0.412 | 0.407 | 0.226 | 0.715 | 0.199 |
| lejepa_aab | 0.773 | 0.770 | 0.671 | 0.923 | 0.514 |
| contrastive_aa | 0.413 | 0.407 | 0.192 | 0.797 | 0.250 |
| contrastive_aab | 0.865 | 0.863 | 0.754 | 0.928 | 0.513 |

## Probes fit on A, applied to C

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa_aa | 0.232 | 0.227 | -0.006 | -2.116 | -0.374 |
| lejepa_aab | 0.537 | 0.542 | 0.198 | 0.729 | -4.697 |
| contrastive_aa | 0.264 | 0.255 | -18.547 | -47.774 | -0.353 |
| contrastive_aab | 0.703 | 0.698 | 0.690 | 0.039 | 0.411 |

## Survey A, good nights only

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa_aa | 0.928 | 0.928 | 0.782 | 0.929 | 0.364 |
| lejepa_aab | 0.896 | 0.895 | 0.770 | 0.939 | 0.626 |
| contrastive_aa | 0.987 | 0.987 | 0.816 | 0.922 | 0.497 |
| contrastive_aab | 0.959 | 0.958 | 0.824 | 0.940 | 0.659 |

## Survey A, bad nights only

| method | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|
| lejepa_aa | 0.767 | 0.759 | 0.525 | 0.864 | 0.164 |
| lejepa_aab | 0.764 | 0.756 | 0.602 | 0.894 | 0.179 |
| contrastive_aa | 0.865 | 0.858 | 0.621 | 0.866 | 0.256 |
| contrastive_aab | 0.804 | 0.798 | 0.659 | 0.882 | 0.407 |

## Survey A (good), MLP probes

| method | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|
| lejepa_aa | 0.944 | 0.944 | 0.824 | 0.965 | 0.603 |
| lejepa_aab | 0.928 | 0.928 | 0.826 | 0.964 | 0.710 |
| contrastive_aa | 0.978 | 0.977 | 0.871 | 0.976 | 0.712 |
| contrastive_aab | 0.965 | 0.965 | 0.864 | 0.972 | 0.754 |

## Survey B (bad), MLP probes

| method | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|
| lejepa_aa | 0.382 | 0.379 | 0.181 | 0.665 | 0.095 |
| lejepa_aab | 0.507 | 0.503 | 0.391 | 0.734 | 0.159 |
| contrastive_aa | 0.390 | 0.386 | 0.181 | 0.688 | 0.061 |
| contrastive_aab | 0.572 | 0.568 | 0.447 | 0.756 | 0.187 |

## Survey C (unseen), MLP probes

| method | cls_acc_mlp | cls_bacc_mlp | logP_r2_mlp | amp_band_r2_mlp | fine_r2_mlp |
|---|---|---|---|---|---|
| lejepa_aa | 0.424 | 0.420 | 0.218 | 0.734 | 0.172 |
| lejepa_aab | 0.822 | 0.821 | 0.777 | 0.946 | 0.564 |
| contrastive_aa | 0.402 | 0.396 | 0.193 | 0.816 | 0.254 |
| contrastive_aab | 0.903 | 0.903 | 0.825 | 0.956 | 0.564 |

## Shared vs private slice (split methods)

| method | slice | cls_acc | cls_bacc | logP_r2 | amp_band_r2 | fine_r2 |
|---|---|---|---|---|---|---|

## Embedding health on A (std_min / std_mean / mean |corr|)

- lejepa_aa: 0.541 / 1.201 / 0.218
- lejepa_aab: 0.579 / 1.166 / 0.224
- contrastive_aa: 0.271 / 0.360 / 0.181
- contrastive_aab: 0.297 / 0.527 / 0.201
