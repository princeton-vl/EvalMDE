import numpy as np

from evalmde.utils.common import uuid, pathlib_file
from evalmde.utils.image import imread_rgb


def get_intermediate_mesh_f(args):
    if args.mesh_dir:
        return args.mesh_dir / f'mesh_{uuid(12)}.glb'
    return args.root / f'mesh_{uuid(12)}.glb'


def get_vis_root(args):
    root = args.root
    valid_triangle_name = 'none'
    if args.valid_triangle_f:
        valid_triangle_name = str((root / args.valid_triangle_f).resolve().relative_to(root.resolve()))
        if args.filter_quad:
            valid_triangle_name = valid_triangle_name + '--filter_quad'
    return pathlib_file(root) / 'visualization' / valid_triangle_name[:-4] / str((root / args.depth_f).resolve().relative_to(root.resolve()))[:-4].replace('/', '_')


def get_crop_region(args):
    if len(args.crop_region) == 0:
        return []
    elif len(args.crop_region) == 4:
        return args.crop_region
    else:
        print(f'Warning: invalid length of crop region (expected 4), {args.crop_region=}. Using [] instead.')
        return []


def get_mesh_vertex_col(args, img_shape):
    '''
    :param args:
    :return: in [0, 1]
    '''
    if getattr(args, 'rgb_f', None):
        rgb = imread_rgb(args.root / args.rgb_f).astype(np.float32) / 255
    else:
        rgb = .7 * np.ones(img_shape + (3,), dtype=np.float32)
        print('no rgb, use gray')
    return rgb


def get_valid_triangle(args, img_shape):
    if getattr(args, 'valid_triangle_f', None):
        ret = np.load(args.root / args.valid_triangle_f)['valid_triangle']
        if args.filter_quad:
            ret[..., 0] &= ret[..., 1]
            ret[..., 1] &= ret[..., 0]
        return ret
    else:
        return np.ones((img_shape[0] - 1, img_shape[1] - 1, 2), dtype=np.bool_)
