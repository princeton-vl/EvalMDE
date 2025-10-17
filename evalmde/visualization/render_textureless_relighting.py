import os
import argparse
import json
from pathlib import Path

import bpy
import mathutils
import numpy as np

from evalmde.utils.image import imread_rgb, resize, imwrite_rgb
from evalmde.utils.proj import apply_SE3
from evalmde.visualization import gen_rot_light__light_pos, ROT_LIGHT_NUM_LIGHT, ROT_LIGHT_NUM_LOOP
from evalmde.visualization.cfg import (get_intermediate_mesh_f, get_vis_root,
                                         get_crop_region, get_mesh_vertex_col, get_valid_triangle)
from evalmde.utils.common import pathlib_file, current_time
from evalmde.utils.depth_to_mesh import gen_mesh_and_pcd
from evalmde.utils.depth import load_data
from evalmde.utils.blender import (bpy_create_cam, bpy_add_ambient_light, bpy_set_tmp_dir, bpy_create_directional_light,
                                     bpy_setup_rgbd_render, bpy_enable_gpu, bpy_render_rgb_and_filter_invalid)


def render(mesh_f, output_root,
           base_cam_pose, cam_intr_params, ds_ratio, num_sample,
           light_i, light_src, overwrite, save_blend,
           ambient, crop_region, cpu):
    cam_pose = base_cam_pose.copy()
    light_src_in_cam = light_src.copy()
    light_src_in_world = apply_SE3(cam_pose, light_src_in_cam)
    light_dst_in_world = apply_SE3(cam_pose, np.array([0, 0, 0.]))

    cam_pose[..., 1:3] *= -1

    output_root = pathlib_file(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    h, w, fx, fy, cx, cy = cam_intr_params

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy_set_tmp_dir(output_root.parent / f'{output_root.name}__tmp')
    if not cpu:
        bpy_enable_gpu()

    assert mesh_f.exists(), mesh_f
    bpy.ops.import_scene.gltf(filepath=str(mesh_f))
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.location = (0, 0, 0)
            obj.scale = (1, 1, 1)
            obj.rotation_mode = 'XYZ'
            obj.rotation_euler = mathutils.Euler((-np.pi / 2, 0, 0), 'XYZ')

    # Set render engine and resolution
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_percentage = 100
    bpy_create_directional_light(light_src_in_world, light_dst_in_world)
    bpy_add_ambient_light(ambient)

    depth_node, rgb_node = bpy_setup_rgbd_render()
    if (not overwrite) and (output_root / f'image_{light_i:06}.png').exists() and \
            (output_root / f'metadata_{light_i:06}.json').exists():
        try:
            with (output_root / f'metadata_{light_i:06}.json').open('r') as F:
                metadata = json.load(F)
            if metadata['num_sample'] == num_sample:
                return
        except Exception as E:
            print(f'{light_i=}, {E=}')

    cam_object = bpy_create_cam(f"cam_{light_i:06}", cam_pose,
                                int(fx), int(fy), int(cx), int(cy), int(w), int(h))
    bpy_render_rgb_and_filter_invalid(cam_object, int(h), int(w), num_sample, depth_node,
                                      rgb_node, str(output_root), f'{light_i:06}', [0, 0, 0], save_depth=False)

    if (output_root / f'image_{light_i:06}.png').exists():
        img = imread_rgb(output_root / f'image_{light_i:06}.png')
        if crop_region is not None and len(crop_region) > 0:
            lb_i, ub_i, lb_j, ub_j = crop_region
            img = img[lb_i:ub_i, lb_j:ub_j]
        img = resize(img, H=ds_ratio * img.shape[0])
        imwrite_rgb(output_root / f'image_{light_i:06}.png', img)
        with (output_root / f'metadata_{light_i:06}.json').open('w') as F:
            json.dump({'num_sample': num_sample, 'time': current_time()}, F)

    if save_blend and light_i == 0:
        out_f = output_root / f'{mesh_f.stem}.blend'
        bpy.ops.wm.save_as_mainfile(filepath=str(out_f))
        print(f'Saved to {out_f.resolve()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--num_sample', type=int, default=256)
    parser.add_argument('--depth_f', type=str, nargs='?', const=None, default='gt_depth.npz', help='Path to depth file, relative to root.')
    parser.add_argument('--valid_triangle_f', type=str, nargs='?', const=None, default='valid_triangle.npz', help='Path to valid triangle file, relative to root.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--save_blend', action='store_true')
    parser.add_argument('--filter_quad', action='store_true', help='Filter out neighboring square if any of triangle is invalid')
    parser.add_argument('--ds_ratio', type=float, default=1)
    parser.add_argument('--ambient', type=float, default=0.2)
    parser.add_argument('--light_l', type=int)
    parser.add_argument('--light_r', type=int)
    parser.add_argument('--crop_region', nargs='*', type=int, default=[], help='Specify 4 integers: lb_i, ub_i, lb_j, ub_j, and only render mesh of [lb_i, ub_i)x[lb_j, ub_j)')
    parser.add_argument('--mesh_dir', type=Path, nargs='?', const=None, default=None)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    root = args.root
    mesh_f = get_intermediate_mesh_f(args)
    vis_root = get_vis_root(args)

    crop_region = get_crop_region(args)
    depth, intr, valid = load_data(root / args.depth_f)
    rgb = get_mesh_vertex_col(args, depth.shape)
    valid_triangle = get_valid_triangle(args, depth.shape)

    mesh, pcd = gen_mesh_and_pcd(intr, depth, valid, rgb=rgb, valid_triangle=valid_triangle, crop_region=crop_region)
    del pcd

    light_pos = gen_rot_light__light_pos(ROT_LIGHT_NUM_LIGHT, ROT_LIGHT_NUM_LOOP)

    mesh_f.parent.mkdir(parents=True, exist_ok=True)
    # mesh.show()
    mesh.export(mesh_f)
    # print(f'Mesh saved to {mesh_f.resolve()}')
    for light_i in range(args.light_l, args.light_r):
        render(mesh_f, vis_root / 'textureless_relighting',
               np.eye(4), list(depth.shape) + intr.tolist(), args.ds_ratio, args.num_sample,
               light_i, light_pos[light_i], args.overwrite, args.save_blend,
               args.ambient, crop_region, args.cpu)
    os.remove(mesh_f)
