import os
import shutil

import bpy
import mathutils
import numpy as np
import OpenEXR
import Imath
from scipy.spatial.transform import Rotation as scipy_Rotation

from evalmde.utils.constants import VALID_DEPTH_LB, VALID_DEPTH_UB
from evalmde.utils.common import pathlib_file
from evalmde.utils.depth import get_depth_valid
from evalmde.utils.image import imread_rgb, imwrite_rgb


def bpy_set_tmp_dir(tmp_dir):
    tmp_dir = pathlib_file(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.temporary_directory = str(tmp_dir)


def bpy_create_cam(cam_name, cam_pose, fx, fy, cx, cy, w, h):
    cam_data = bpy.data.cameras.new(name=cam_name)
    cam_data.sensor_height = cam_data.sensor_width * h / w
    cam_data.lens = (fx + fy) / 2 * cam_data.sensor_width / w
    cam_data.shift_x = (w / 2 - cx) / w
    cam_data.shift_y = (cy - h / 2) / h

    cam_object = bpy.data.objects.new(cam_name, cam_data)
    bpy.context.collection.objects.link(cam_object)
    cam_object.matrix_world = mathutils.Matrix([cam_pose[0], cam_pose[1], cam_pose[2], cam_pose[3]])
    return cam_object


def bpy_add_ambient_light(energy=1.0):
    world = bpy.data.worlds.new("AmbientWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (1, 1, 1, 1)
    bg.inputs[1].default_value = energy


def bpy_enable_gpu(device_type="CUDA"):
    prefs = bpy.context.preferences
    cprefs = prefs.addons['cycles'].preferences
    cprefs.compute_device_type = device_type  # "CUDA", "OPTIX", "METAL", "HIP"
    cprefs.get_devices()  # Initialize devices
    for device in cprefs.devices:
        device.use = True
    bpy.context.scene.cycles.device = 'GPU'


def bpy_setup_rgb_render():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    rgb_output = tree.nodes.new(type='CompositorNodeOutputFile')
    rgb_output.label = 'RGB Output'
    rgb_output.base_path = ''
    rgb_output.format.file_format = 'PNG'
    rgb_output.file_slots[0].use_node_format = True
    rgb_output.file_slots[0].save_as_render = True
    tree.links.new(render_layers.outputs['Image'], rgb_output.inputs[0])
    return rgb_output


def bpy_setup_depth_render():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')
    depth_output = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output.label = 'Depth Output'
    depth_output.base_path = ''
    depth_output.format.file_format = 'OPEN_EXR'
    depth_output.file_slots[0].use_node_format = True
    depth_output.file_slots[0].save_as_render = True
    bpy.context.view_layer.use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True
    tree.links.new(render_layers.outputs['Depth'], depth_output.inputs[0])
    return depth_output


def bpy_setup_rgbd_render():
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    tree.nodes.clear()

    # Add Render Layers node to get passes
    render_layers = tree.nodes.new(type='CompositorNodeRLayers')

    # Add Output File node to save the EXR
    depth_output = tree.nodes.new(type='CompositorNodeOutputFile')
    depth_output.label = 'Depth Output'
    depth_output.base_path = ''
    depth_output.format.file_format = 'OPEN_EXR'
    depth_output.file_slots[0].use_node_format = True
    depth_output.file_slots[0].save_as_render = True

    # Output for RGB
    rgb_output = tree.nodes.new(type='CompositorNodeOutputFile')
    rgb_output.label = 'RGB Output'
    rgb_output.base_path = ''
    rgb_output.format.file_format = 'PNG'
    rgb_output.file_slots[0].use_node_format = True
    rgb_output.file_slots[0].save_as_render = True

    # Enable the Z (Depth) pass
    bpy.context.view_layer.use_pass_z = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

    # Link the depth pass output from the render layers node
    tree.links.new(render_layers.outputs['Depth'], depth_output.inputs[0])
    tree.links.new(render_layers.outputs['Image'], rgb_output.inputs[0])

    return depth_output, rgb_output


def save_depth_from_exr(filepath, h, w):
    exr_file = OpenEXR.InputFile(filepath)

    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    assert size == (w, h), f"Expected {(w, h)}, got {size}"

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = exr_file.channels(["R", "G", "B"], pt)
    depth = [np.frombuffer(c, dtype=np.float32).reshape(size[1], size[0]) for c in channels]
    assert np.all(depth[0] == depth[1]) and np.all(depth[0] == depth[2])
    return depth[0]


def bpy_create_directional_light(src: np.ndarray, dst: np.ndarray, energy=5.0, name='Sun'):
    light_data = bpy.data.lights.new(name=name, type='SUN')
    light_data.energy = energy
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (float(src[0]), float(src[1]), float(src[2]))

    direction = dst - src
    rot_axis = np.cross(np.array([0, 0, -1.]), direction)
    if np.linalg.norm(rot_axis) < 1e-5:
        rot_axis = np.array([1., 0, 0])
    rot_axis /= np.linalg.norm(rot_axis)
    rot_ang = np.arccos(np.clip(((direction / np.linalg.norm(direction)) * np.array([0, 0, -1.])).sum(), -1, 1))
    rot_euler = scipy_Rotation.from_rotvec(rot_ang * rot_axis, degrees=False).as_euler('xyz', degrees=False)
    light_obj.rotation_euler = (float(rot_euler[0]), float(rot_euler[1]), float(rot_euler[2]))


def bpy_render_rgb(cam_object, h, w, num_sample, rgb_node, output_root, out_name):
    bpy.context.scene.cycles.samples = num_sample
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h

    bpy.context.scene.camera = cam_object
    rgb_node.base_path = str(output_root)
    rgb_node.file_slots[0].path = f"image_{out_name}-"
    bpy.ops.render.render(write_still=True)


def bpy_render_rgbd(cam_object, h, w, num_sample, depth_node, rgb_node, output_root, out_name):
    bpy.context.scene.cycles.samples = num_sample
    bpy.context.scene.render.resolution_x = w
    bpy.context.scene.render.resolution_y = h

    bpy.context.scene.camera = cam_object
    depth_node.base_path = str(output_root)
    depth_node.file_slots[0].path = f"depth_{out_name}_"
    rgb_node.base_path = str(output_root)
    rgb_node.file_slots[0].path = f"image_{out_name}-"
    bpy.ops.render.render(write_still=True)
    exr_path = os.path.join(str(output_root), f"depth_{out_name}_0001.exr")
    depth_np = save_depth_from_exr(exr_path, h, w)
    np.save(os.path.join(str(output_root), f"depth_{out_name}.npy"), depth_np)
    os.remove(exr_path)


def bpy_render_rgb_and_filter_invalid(cam_object, h, w, num_sample, depth_node, rgb_node, output_root, out_name, bkg_color, valid_depth_lb=VALID_DEPTH_LB, valid_depth_ub=VALID_DEPTH_UB, save_depth=False):
    '''
    :param cam_object:
    :param h:
    :param w:
    :param num_sample:
    :param depth_node:
    :param rgb_node:
    :param output_root:
    :param out_name:
    :param bkg_color: list of 3 integers, [0, 255]
    :param valid_depth_lb:
    :param valid_depth_ub:
    :param save_depth:
    :return:
    '''
    bpy_render_rgbd(cam_object, h, w, num_sample, depth_node, rgb_node, output_root, out_name)

    img_f = pathlib_file(os.path.join(str(output_root), f"image_{out_name}-0001.png"))
    depth_f = pathlib_file(os.path.join(str(output_root), f"depth_{out_name}.npy"))
    output_root = pathlib_file(output_root)
    tmp_dir = output_root.parent / f'{output_root.name}__tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    img = imread_rgb(img_f)
    depth = np.load(depth_f)
    shutil.move(img_f, tmp_dir / img_f.name)
    if not save_depth:
        shutil.move(depth_f, tmp_dir / depth_f.name)
    img[~get_depth_valid(depth, valid_depth_lb, valid_depth_ub)] = np.array(bkg_color)
    imwrite_rgb(output_root / f'image_{out_name}.png', img)
    os.remove(tmp_dir / img_f.name)
    os.remove(tmp_dir / depth_f.name)
