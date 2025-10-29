from typing import Tuple

import numpy as np
import torch

from evalmde.utils.constants import VALID_DEPTH_LB, VALID_DEPTH_UB
from evalmde.utils.torch import reformat_as_torch_tensor


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    # https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8/src/util/alignment.py#L8
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def align_affine_lstsq(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    # https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/moge/utils/alignment.py#L399
    """
    Solve `min sum_i w_i * (a * x_i + b - y_i ) ^ 2`, where `a` and `b` are scalars, with respect to `a` and `b` using least squares.

    ### Parameters:
    - `x: torch.Tensor` of shape (..., N)
    - `y: torch.Tensor` of shape (..., N)
    - `w: torch.Tensor` of shape (..., N)

    ### Returns:
    - `a: torch.Tensor` of shape (...,)
    - `b: torch.Tensor` of shape (...,)
    """
    w_sqrt = torch.ones_like(x) if w is None else w.sqrt()
    A = torch.stack([w_sqrt * x, torch.ones_like(x)], dim=-1)
    B = (w_sqrt * y)[..., None]
    a, b = torch.linalg.lstsq(A, B)[0].squeeze(-1).unbind(-1)
    return a, b


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

    raise NotImplementedError(f'{method=}')
