import torch
import torch.nn.functional as F


'''
VERTEX_SLICES:
0 2
1 3
'''
VERTEX_SLICES = [
    (slice(None, -1), slice(None, -1)),
    (slice(1, None), slice(None, -1)),
    (slice(None, -1), slice(1, None)),
    (slice(1, None), slice(1, None)),
]
TRIANGLE_SLICES = [
    [VERTEX_SLICES[0], VERTEX_SLICES[1], VERTEX_SLICES[2]],
    [VERTEX_SLICES[2], VERTEX_SLICES[0], VERTEX_SLICES[3]],
    [VERTEX_SLICES[0], VERTEX_SLICES[1], VERTEX_SLICES[3]],
    [VERTEX_SLICES[2], VERTEX_SLICES[1], VERTEX_SLICES[3]],
]
NUM_TRIANGLE = len(TRIANGLE_SLICES)


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
    ret = torch.empty((H - 1, W - 1, NUM_TRIANGLE), dtype=torch.bool, device=device)
    for i, TRIANGLE_SLICE in enumerate(TRIANGLE_SLICES):
        ret[..., i] = _fetch_pixel_val(valid, TRIANGLE_SLICE[0]) & \
                      _fetch_pixel_val(valid, TRIANGLE_SLICE[1]) & \
                      _fetch_pixel_val(valid, TRIANGLE_SLICE[2])
    return ret


@torch.no_grad()
def get_triangle_normal(xyz: torch.Tensor):
    '''
    :param xyz: shape (H, W, 3)
    :return: normal, normal_valid
        normal: shape (H - 1, W - 1, NUM_TRIANGLE, 3)
        normal_valid: shape (H - 1, W - 1, NUM_TRIANGLE)
    '''
    H, W = xyz.shape[:2]
    device = xyz.device
    dtype = xyz.dtype
    normal = torch.empty((H - 1, W - 1, NUM_TRIANGLE, 3), dtype=dtype, device=device)
    normal_valid = torch.empty((H - 1, W - 1, NUM_TRIANGLE), dtype=torch.bool, device=device)
    for i, TRIANGLE_SLICE in enumerate(TRIANGLE_SLICES):
        normal[..., i, :] = torch.linalg.cross(
            F.normalize(_fetch_pixel_val(xyz, TRIANGLE_SLICE[1]) - _fetch_pixel_val(xyz, TRIANGLE_SLICE[0]), dim=-1),
            F.normalize(_fetch_pixel_val(xyz, TRIANGLE_SLICE[2]) - _fetch_pixel_val(xyz, TRIANGLE_SLICE[0]), dim=-1),
            dim=-1
        )
        vec_norm = torch.norm(normal[..., i, :], dim=-1)  # (H - 1, W - 1)
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
