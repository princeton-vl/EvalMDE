from evalmde.utils.depth import load_data
# gt_depth, gt_intr, gt_valid = load_data('sample_data/gt_depth.npz')
# pr_depth, pr_intr, pr_valid = load_data('sample_data/curv_low_freq__0.200_10.0.npz')
gt_depth, gt_intr, gt_valid = load_data('sample_data_2/gt_depth.npz')
pr_depth, pr_intr, pr_valid = load_data('sample_data_2/depthpro_gt_focal.npz')


from evalmde.metrics.rel_normal import compute_rel_normal
from evalmde.metrics.sawa_h import compute_sawa_h
sawa_h = compute_sawa_h(pr_depth, pr_intr, pr_valid, gt_depth, gt_intr, gt_valid)
rel_normal = compute_rel_normal(pr_depth, pr_intr, pr_valid, gt_depth, gt_intr, gt_valid)
print(f'{sawa_h=}, {rel_normal=}')
