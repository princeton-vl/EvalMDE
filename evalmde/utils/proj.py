import numpy as np
import torch
import torch.nn.functional as F


def th_uv_grid(H: int, W: int, **tensor_kwargs) -> torch.Tensor:
    '''
    :param H: int
    :param W: int
    :param tensor_kwargs:
    :return: (H, W, 2)
    '''
    v, u = torch.meshgrid(torch.arange(H).to(**tensor_kwargs), torch.arange(W).to(**tensor_kwargs))
    return torch.stack([u, v], dim=-1)


def depth_to_xyz(intr, depth):
    '''
    :param intr: shape (4,)
    :param depth: shape (H, W)
    :return: shape (H, W, 3)
    '''
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    if isinstance(depth, np.ndarray):
        v, u = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        return np.stack([x, y, depth], axis=-1)
    elif isinstance(depth, torch.Tensor):
        tensor_kwargs = dict(device=depth.device, dtype=depth.dtype)
        v, u = torch.meshgrid(torch.arange(depth.shape[0]).to(**tensor_kwargs), torch.arange(depth.shape[1]).to(**tensor_kwargs))
        x = (u - cx) / fx * depth
        y = (v - cy) / fy * depth
        return torch.stack([x, y, depth], dim=-1)
    else:
        raise ValueError(f'{type(depth)=}')


def apply_SE3(SE3, pnt):
    assert SE3.shape == (4, 4) and pnt.shape[-1] == 3
    return (SE3[:3, :3] @ pnt[..., None])[..., 0] + SE3[:3, -1]
