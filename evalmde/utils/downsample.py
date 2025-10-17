from typing import List

import torch
import torch.nn.functional as F

from evalmde.utils.proj import th_uv_grid


def pad(x: torch.Tensor, sc: int) -> torch.Tensor:
    '''
    pad x to bottom and right with 0, so that H % sc == 0 and W % sc == 0
    :param x: shape (H, W, ...)
    :param sc: int
    :return: pad_x
    '''
    H, W, C_shape = x.shape[0], x.shape[1], x.shape[2:]
    x = x.reshape(H, W, -1).permute(2, 0, 1)  # (-1, H, W)
    pad_H = (sc - H % sc) % sc
    pad_W = (sc - W % sc) % sc
    x = F.pad(x, (0, pad_W, 0, pad_H), value=0)  # (-1, H', W')
    return x.permute(1, 2, 0).reshape((x.shape[-2], x.shape[-1]) + C_shape)


def patchify(x: torch.Tensor, sc: int):
    '''
    reshape (H, W, ...) to (sc, sc, H / sc, W / sc, ...)
    :param x: shape (H, W, ...)
    :param sc: int
    :return: patched_x
    '''
    H, W, C_shape = x.shape[0], x.shape[1], x.shape[2:]
    assert H % sc == 0 and W % sc == 0, f'can\'t patchify ({x.shape=}, {sc=})'
    _H, _W = H // sc, W // sc
    x = x.reshape(_H, sc, _W, sc, -1).permute(1, 3, 0, 2, 4)
    return x.reshape((sc, sc, _H, _W) + C_shape)


def gather(x: torch.Tensor, idx: torch.Tensor):
    '''
    :param x: shape (sc, sc, H / sc, W / sc, ...)
    :param idx: shape (H / sc, W / sc)
    :return: x[idx[i,j] // sc, idx[i,j] % sc, i, j, ...]
    '''
    sc, _, H, W, C_shape = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4:]
    x = x.reshape(sc * sc, H, W, -1)
    idx = idx[None, :, :, None].repeat(1, 1, 1, x.shape[-1])  # (1, H / sc, W / sc, -1)
    return torch.gather(x, 0, idx).reshape((H, W) + C_shape)


def downsample(ds_sc: int, valid: torch.Tensor, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    '''
    :param ds_sc: downsample scale
    :param valid: (H, W), dtype: torch.bool
    :param tensors: list of tensors of shape (H, W, ...)
    :return: [ds_valid, *ds_tensors]
        ds_valid: (ds_H, ds_W)
        ds_tensors: list of tensors of shape (ds_H, ds_W, ...)
    '''
    tensor_kwargs = dict(device=valid.device, dtype=torch.float)
    H, W = valid.shape
    uv = th_uv_grid(H, W, **tensor_kwargs)  # (H, W, 2)
    uv = patchify(pad(uv, ds_sc), ds_sc)  # (sc, sc, H / sc, W / sc, 2)
    ds_H, ds_W = uv.shape[2], uv.shape[3]
    patch_center = th_uv_grid(ds_H, ds_W, **tensor_kwargs) * ds_sc + .5 * (ds_sc - 1)  # (H / sc, W / sc, 2)
    valid = patchify(pad(valid, ds_sc), ds_sc)  # (sc, sc, H / sc, W / sc)
    uv_dst = (uv - patch_center[None, None]).norm(dim=-1)  # (sc, sc, H / sc, W / sc)
    uv_dst[~valid] = torch.inf  # mask out invalid pixels
    uv_dst = uv_dst.reshape(-1, uv_dst.shape[-2], uv_dst.shape[-1])  # (sc * sc, H / sc, W / sc)
    ds_pxl = torch.argmin(uv_dst, dim=0)  # (H / sc, W / sc)
    valid = gather(valid, ds_pxl)
    tensors = [gather(patchify(pad(x, ds_sc), ds_sc), ds_pxl) for x in tensors]
    return [valid] + tensors
