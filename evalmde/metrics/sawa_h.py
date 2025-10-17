import numpy as np

from evalmde.utils.proj import depth_to_xyz
from evalmde.utils.depth import align
from evalmde.utils.torch import reformat_as_torch_tensor
from evalmde.metrics.standard import rel_depth, delta0125
from evalmde.metrics.boundary import boundary_f1
from evalmde.metrics.rel_normal import rel_normal as rel_normal_func


def compute_sawa_h(pred_depth: np.ndarray, pred_intr: np.ndarray, pred_valid: np.ndarray,
                   gt_depth: np.ndarray, gt_intr: np.ndarray, gt_valid: np.ndarray) -> float:
    '''
    :param pred_depth: shape (H, W)
    :param pred_intr: shape (4,), [fx, fy, cx, cy]
    :param pred_valid: shape (H, W), dtype: np.bool_
    :param gt_depth: shape (H, W)
    :param gt_intr: shape (4,), [fx, fy, cx, cy]
    :param gt_valid: shape (H, W), dtype: np.bool_
    :return: SAWA-H value
    '''
    wkdr__no_align = 1 - rel_depth(pred_depth, gt_depth, gt_valid)[1]
    delta0125__disparity_af_clip_by_0 = 1 - delta0125(align(
        1 / reformat_as_torch_tensor(pred_depth),
        reformat_as_torch_tensor(gt_depth),
        reformat_as_torch_tensor(gt_valid),
        'disparity_affine_clip_by_0'
    ), gt_depth, gt_valid)[1]
    delta0125__depth_af_lst_sq_clip_by_0 = 1 - delta0125(align(
        reformat_as_torch_tensor(pred_depth),
        reformat_as_torch_tensor(gt_depth),
        reformat_as_torch_tensor(gt_valid),
        'depth_affine_lst_sq_clip_by_0'
    ), gt_depth, gt_valid)[1]
    boundary__no_align = boundary_f1(
        reformat_as_torch_tensor(pred_depth),
        reformat_as_torch_tensor(gt_depth),
        reformat_as_torch_tensor(gt_valid)
    )[1]
    rel_normal = rel_normal_func(
        depth_to_xyz(gt_intr, gt_depth), gt_valid,
        depth_to_xyz(pred_intr, pred_depth), pred_valid,
    )
    err = 3.65 * wkdr__no_align + 0.18 * delta0125__disparity_af_clip_by_0 + 0.01 * delta0125__depth_af_lst_sq_clip_by_0 + 0.20 * boundary__no_align + 1.94 * rel_normal
    return err
