from pathlib import Path
gt_depth_f = Path('sample_data_2/gt_depth.npz')
valid_triangle_f = Path('sample_data_2/valid_triangle.npz')


THRESH = 1.1

import numpy as np
def induce_valid_triangle_from_gt_depth(gt_depth: np.ndarray, valid: np.ndarray):
    '''
    :param gt_depth: shape (H, W)
    :param valid: shape (H, W)
    :return: valid_triangle, shape (H - 1, W - 1, 2)
    '''
    min_d_0 = np.min(np.stack([gt_depth[:-1, :-1], gt_depth[1:, :-1], gt_depth[:-1, 1:]], axis=0), axis=0)
    max_d_0 = np.max(np.stack([gt_depth[:-1, :-1], gt_depth[1:, :-1], gt_depth[:-1, 1:]], axis=0), axis=0)
    valid_0 = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & (max_d_0 <= THRESH * min_d_0)

    min_d_1 = np.min(np.stack([gt_depth[1:, 1:], gt_depth[1:, :-1], gt_depth[:-1, 1:]], axis=0), axis=0)
    max_d_1 = np.max(np.stack([gt_depth[1:, 1:], gt_depth[1:, :-1], gt_depth[:-1, 1:]], axis=0), axis=0)
    valid_1 = valid[1:, 1:] & valid[:-1, 1:] & valid[1:, :-1] & (max_d_1 <= THRESH * min_d_1)
    return np.stack([valid_0, valid_1], axis=-1)


from evalmde.utils.depth import load_data
gt_depth, gt_intr, gt_valid = load_data(gt_depth_f)
valid_triangle = induce_valid_triangle_from_gt_depth(gt_depth, gt_valid)
np.savez(valid_triangle_f, valid_triangle=valid_triangle)
print(f'Saved to {valid_triangle_f.resolve()}')
