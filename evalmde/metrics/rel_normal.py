from typing import List
from math import floor

import numpy as np
import torch
import torch.nn.functional as F

from evalmde.utils.torch import get_angle_between, reformat_as_torch_tensor
from evalmde.utils.downsample import downsample
from evalmde.utils.proj import depth_to_xyz

DEFAULT_CONFIG={
    'scales': [1, 2, 4, 8],
    'num_sample': int(1e6),
    'radius': 32,
    'min_radius': 3,
    'invalid': 'penalty',
}

@torch.no_grad()
def _fetch_pixel_val(x: torch.Tensor, vertex_slice):
    '''
    :param x: shape (H, W, ...)
    :param vertex_slice:
    :return: shape (H - 1, W - 1, ...)
    '''
    return x[vertex_slice[0], vertex_slice[1]]


@torch.no_grad()
def get_triangle_valid(valid: torch.Tensor):
    '''
    a triangle is valid if all vertices are valid
    :param valid: shape (H, W)
    :return: triangle_valid
        triangle_valid: shape (H - 1, W - 1, NUM_TRIANGLE)
    '''
    H, W = valid.shape
    device = valid.device
    ret = torch.empty((H - 2, W - 2, NUM_TRIANGLE), dtype=torch.bool, device=device)
    for i, TRIANGLE_SLICE in enumerate(TRIANGLE_SLICES):
        ret[..., i] = _fetch_pixel_val(valid, TRIANGLE_SLICE[0]) & \
                      _fetch_pixel_val(valid, TRIANGLE_SLICE[1]) & \
                      _fetch_pixel_val(valid, TRIANGLE_SLICE[2])
    return ret

TRIANGLE_SLICES=((
    (slice(None, -2), slice(None, -2)),
    (slice(2, None), slice(None, -2)),
    (slice(None, -2), slice(2, None)),
),)
NUM_TRIANGLE = 1
@torch.no_grad()
def get_triangle_normal(xyz: torch.Tensor):
    '''
    Normal computation method 2: 2-pixel spacing
    :param xyz: shape (H, W, 3)
    :return: normal, normal_valid
        normal: shape (H - 2, W - 2, NUM_TRIANGLE_2, 3)
        normal_valid: shape (H - 2, W - 2, NUM_TRIANGLE_2)
    '''
    H, W = xyz.shape[:2]
    device = xyz.device
    dtype = xyz.dtype
    normal = torch.empty((H - 2, W - 2, 1, 3), dtype=dtype, device=device)
    normal_valid = torch.empty((H - 2, W - 2, 1), dtype=torch.bool, device=device)
    for i, TRIANGLE_SLICE in enumerate(TRIANGLE_SLICES):
        normal[..., i, :] = torch.linalg.cross(
            F.normalize(_fetch_pixel_val(xyz, TRIANGLE_SLICE[1]) - _fetch_pixel_val(xyz, TRIANGLE_SLICE[0]), dim=-1),
            F.normalize(_fetch_pixel_val(xyz, TRIANGLE_SLICE[2]) - _fetch_pixel_val(xyz, TRIANGLE_SLICE[0]), dim=-1),
            dim=-1
        )
        vec_norm = torch.norm(normal[..., i, :], dim=-1)
        normal_valid[..., i] = vec_norm > 1e-5
        normal[..., i, :] /= vec_norm.clamp(min=1e-5).unsqueeze(-1)
    return normal, normal_valid

@torch.no_grad()
def get_triangle_normal_and_valid(xyz: torch.Tensor, valid: torch.Tensor, flatten: bool = True):
    '''
    if gt_d and depth_layer are not None, filter out triangle across depth layers
    :param xyz:
    :param valid:
    :param flatten:
    :return: normal, valid
    '''
    normal, normal_valid = get_triangle_normal(xyz)
    tri_valid = get_triangle_valid(valid)
    valid = normal_valid & tri_valid
    if flatten:
        normal = normal.reshape(-1, 3)
        valid = valid.reshape(-1)
    return normal, valid


@torch.no_grad()
def get_angle_between(n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
    '''
    :param n1: shape (..., 3), norm > 0
    :param n2: shape (..., 3), norm > 0
    :return: shape (...)
    '''
    return torch.acos((F.normalize(n1, dim=-1) * F.normalize(n2, dim=-1)).sum(dim=-1).clamp(-1, 1))

@torch.no_grad()
def get_pair_pxl(H: int, W: int, num_sample: int, radius: int, device):
    radius = min(radius, max(H, W))
    i1 = torch.empty((num_sample,), dtype=torch.long, device=device)
    j1 = torch.empty((num_sample,), dtype=torch.long, device=device)
    i2 = torch.empty((num_sample,), dtype=torch.long, device=device)
    j2 = torch.empty((num_sample,), dtype=torch.long, device=device)

    n = 0
    s = torch.quasirandom.SobolEngine(4)
    while n < num_sample:
        samples = s.draw(floor(num_sample * 1.1)).to(device)
        samples[:,0] *= H
        samples[:,1] *= W
        samples[:,2] *= radius * 2
        samples[:,2] -= radius
        samples[:,3] *= radius * 2
        samples[:,3] -= radius
        points = torch.cat([samples[:,:2], samples[:,:2] + samples[:,2:]], dim=1)
        points = torch.floor(points)

        valid = (points[:,[0,2]] < H).all(dim=-1) & (points[:,[1,3]] < W).all(dim=-1) & (0 <= points[:,[0,2]]).all(dim=-1) & (0 <= points[:,[1,3]]).all(dim=-1)
        points = points[valid]
        m = min(len(points), num_sample - n)
        i1[n:n+m] = points[:m,0]
        j1[n:n+m] = points[:m,1]
        i2[n:n+m] = points[:m,2]
        j2[n:n+m] = points[:m,3]
        n += m
    
    return i1, j1, i2, j2


@torch.no_grad()
def get_rel_normal_err_heatmap_idx(gt_xyz: torch.Tensor, gt_valid: torch.Tensor,
                               pred_xyz: torch.Tensor, pred_valid: torch.Tensor,
                               num_sample: int, radius: int):
    '''
    :param gt_xyz:
    :param gt_valid:
    :param pred_xyz:
    :param pred_valid:
    :param num_sample:
    :param radius:
    :return: rel_normal_err, gt_pair_valid, pred_pair_valid
        rel_normal_err: shape (-1,)
        gt_pair_valid: shape (-1,)
        pred_pair_valid: shape (-1,)
    '''
    gt_normal, gt_normal_valid = get_triangle_normal_and_valid(gt_xyz, gt_valid, flatten=False)
    pred_normal, pred_normal_valid = get_triangle_normal_and_valid(pred_xyz, pred_valid, flatten=False)

    H, W = gt_normal.shape[:2]
    i1, j1, i2, j2 = get_pair_pxl(H, W, num_sample, radius, gt_xyz.device)

    gt_rel_normal = get_angle_between(gt_normal[i1, j1], gt_normal[i2, j2])
    gt_pair_valid = gt_normal_valid[i1, j1] & gt_normal_valid[i2, j2]
    pred_rel_normal = get_angle_between(pred_normal[i1, j1], pred_normal[i2, j2])
    pred_pair_valid = pred_normal_valid[i1, j1] & pred_normal_valid[i2, j2]
    rel_normal_err = torch.abs(gt_rel_normal - pred_rel_normal)  # [0, pi]
    return rel_normal_err, gt_pair_valid, pred_pair_valid, (i1,j1,i2,j2)



def get_multi_scale_rel_normal_err(gt_xyz: torch.Tensor, gt_valid: torch.Tensor,
                                   pred_xyz: torch.Tensor, pred_valid: torch.Tensor,
                                   scales: List[int], num_sample: int, radius: int, min_radius: int, invalid):
    '''
    :param gt_xyz:
    :param gt_valid:
    :param pred_xyz:
    :param pred_valid:
    :param scales: list of down-sample scales
    :param num_sample:
    :param radius:
    :param min_radius:
    :return: list of avg relative normal errors under each scale
    '''
    ret = []
    for sc in scales:
        ds_gt_valid, ds_gt_xyz, ds_pred_valid, ds_pred_xyz = downsample(sc, gt_valid, [gt_xyz, pred_valid, pred_xyz])
        err, gt_pair_valid, pred_pair_valid, _ = get_rel_normal_err_heatmap_idx(ds_gt_xyz, ds_gt_valid, ds_pred_xyz, ds_pred_valid, num_sample, max(radius // sc, min_radius))
        match invalid:
            case 'penalty':
                err = torch.where(gt_pair_valid & ~pred_pair_valid, torch.pi, err)
                err = err[gt_pair_valid]
            case 'ignore':
                err = err[gt_pair_valid & pred_pair_valid]
            case _:
                raise ValueError()

        if err.shape[0] > 0:
            scalar_err = err.mean().item()
            ret.append(scalar_err)
    if len(ret) == 0:
        ret = [0]
    return ret


def rel_normal(gt_xyz, gt_valid, pred_xyz, pred_valid, cfg=None, **kwargs):
    if cfg is None:
        cfg = DEFAULT_CONFIG | kwargs
    device_args = {k:v for k,v in cfg.items() if k == 'device'}
    cfg.pop('device', None)
    gt_xyz = reformat_as_torch_tensor(gt_xyz, **device_args)
    gt_valid = reformat_as_torch_tensor(gt_valid, **device_args)
    pred_xyz = reformat_as_torch_tensor(pred_xyz, **device_args)
    pred_valid = reformat_as_torch_tensor(pred_valid, **device_args)
    return np.mean(get_multi_scale_rel_normal_err(gt_xyz, gt_valid, pred_xyz, pred_valid, **cfg))


def compute_rel_normal(pred_depth: np.ndarray, pred_intr: np.ndarray, pred_valid: np.ndarray,
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
    err = rel_normal(
        depth_to_xyz(gt_intr, gt_depth), gt_valid,
        depth_to_xyz(pred_intr, pred_depth), pred_valid,
    )
    return err
