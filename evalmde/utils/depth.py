import sys

import numpy as np
import torch

from evalmde import MOGE_PATH, MARIGOLD_PATH
from evalmde.utils.constants import VALID_DEPTH_LB, VALID_DEPTH_UB
from evalmde.utils.torch import reformat_as_torch_tensor

sys.path.append(str(MOGE_PATH))
from moge.utils.geometry_torch import mask_aware_nearest_resize
from moge.utils.alignment import (
    align_points_scale_z_shift,
    align_points_scale_xyz_shift,
    align_points_xyz_shift,
    align_affine_lstsq,
    align_depth_scale,
    align_depth_affine,
    align_points_scale,
)

sys.path.append(str(MARIGOLD_PATH))
from src.util.alignment import align_depth_least_square


def get_depth_valid(depth, valid_depth_lb=VALID_DEPTH_LB, valid_depth_ub=VALID_DEPTH_UB):
    if isinstance(depth, np.ndarray):
        return (~np.isnan(depth)) & (~np.isinf(depth)) & (depth >= valid_depth_lb) & (depth <= valid_depth_ub)
    elif isinstance(depth, torch.Tensor):
        return (~torch.isnan(depth)) & (~torch.isinf(depth)) & (depth >= valid_depth_lb) & (depth <= valid_depth_ub)
    else:
        raise ValueError(f'{type(depth)=}')


def load_data(depth_f, as_torch=False):
    data = np.load(depth_f)
    depth, intr, valid = data['depth'], data['intr'], data['valid']
    depth[~valid] = 1
    if as_torch:
        depth = reformat_as_torch_tensor(depth)
        intr = reformat_as_torch_tensor(intr)
        valid = reformat_as_torch_tensor(valid)
    return depth, intr, valid


def align(pred, gt, gt_valid, method, return_align_param=False, eps=1e-4):
    if method == 'no':
        if return_align_param:
            return pred, None
        return pred
    if method == 'depth_scale-no_weight':
        # pred: scale-invariant depth
        # gt: gt depth
        # return: aligned depth
        if not gt_valid.any():
            sc = 1
        else:
            sc = (gt[gt_valid] * pred[gt_valid]).sum() / (pred[gt_valid] * pred[gt_valid]).sum()
        if return_align_param:
            return sc * pred, float(sc)
        return sc * pred

    if method == 'depth_affine_lst_sq_clip_by_0':
        # pred: affine-invariant depth
        # gt: gt depth
        # return: aligned depth
        ret, scale, shift = align_depth_least_square(gt.cpu().numpy(), pred.cpu().numpy(), gt_valid.cpu().numpy())
        ret = torch.from_numpy(ret).to(device=pred.device, dtype=pred.dtype).clamp_min(eps)
        if return_align_param:
            return ret, (float(scale), float(shift))
        return ret

    if method in ['disparity_affine', 'disparity_affine_clip_by_0']:
        # pred: predicted affine-invariant disparity
        # gt: gt depth
        # return: aligned depth
        scale, shift = align_affine_lstsq(pred[gt_valid], 1 / gt[gt_valid])
        pred_disp = pred * scale + shift
        if method == 'disparity_affine':
            ret = 1 / pred_disp.clamp_min(1 / gt[gt_valid].max().item())
        else:
            ret = 1 / pred_disp.clamp_min(eps)
        if return_align_param:
            return ret, (float(scale), float(shift))
        return ret

    _, lr_mask, lr_index = mask_aware_nearest_resize(None, gt_valid, (64, 64), return_index=True)
    pred_lr_masked, gt_lr_masked = pred[lr_index][lr_mask], gt[lr_index][lr_mask]
    if method == 'depth_scale':
        # pred: scale-invariant depth
        # gt: gt depth
        # return: aligned depth
        scale = align_depth_scale(pred_lr_masked, gt_lr_masked, 1 / gt_lr_masked)
        if return_align_param:
            return scale * pred, float(scale)
        return scale * pred
    elif method in ['depth_affine', 'depth_affine_clip_by_0']:
        # pred: affine-invariant depth
        # gt: gt depth
        # return: aligned depth
        scale, shift = align_depth_affine(pred_lr_masked, gt_lr_masked, 1 / gt_lr_masked)
        ret = scale * pred + shift
        if method == 'depth_affine_clip_by_0':
            ret = ret.clamp_min(eps)
        if return_align_param:
            return ret, (float(scale), float(shift))
        return ret
    elif method == 'point_scale':
        # pred: scale-invariant point map
        # gt: gt point map
        # return: aligned point map
        scale = align_points_scale(pred_lr_masked, gt_lr_masked, 1 / gt_lr_masked.norm(dim=-1))
        if return_align_param:
            return scale * pred, float(scale)
        return scale * pred
    elif method == 'point_affine':
        # pred: affine-invariant point map
        # gt: gt point map
        # return: aligned point map
        scale, shift = align_points_scale_xyz_shift(pred_lr_masked, gt_lr_masked, 1 / gt_lr_masked.norm(dim=-1))
        if return_align_param:
            return scale * pred + shift, (float(scale), shift)
        return scale * pred + shift
    else:
        raise ValueError(f'{method=}')
