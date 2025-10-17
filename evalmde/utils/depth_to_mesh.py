import numpy as np
import open3d as o3d
import trimesh

from evalmde.utils.proj import depth_to_xyz, apply_SE3


def gen_triangle_v_idx(H, W):
    pxl_idx = np.arange(H * W).reshape(H, W)
    triangle_v_idx = np.stack([
        np.stack([pxl_idx[:-1, :-1], pxl_idx[1:, :-1], pxl_idx[:-1, 1:]], axis=-1),  # (H - 1, W - 1, 3)
        np.stack([pxl_idx[1:, 1:], pxl_idx[:-1, 1:], pxl_idx[1:, :-1]], axis=-1),  # (H - 1, W - 1, 3)
    ], axis=-2)  # (H - 1, W - 1, 2, 3)
    return triangle_v_idx


def gen_trimesh_mesh(vs, cs, triangles):
    mesh_vertices = o3d.utility.Vector3dVector(vs.reshape(-1, 3))
    mesh_faces = o3d.utility.Vector3iVector(triangles)
    mesh = o3d.geometry.TriangleMesh(mesh_vertices, mesh_faces)
    mesh.compute_vertex_normals()

    trimesh_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_normals=np.asarray(mesh.vertex_normals),
        vertex_colors=cs.reshape(-1, 3),
        process=False
    )
    material = trimesh.visual.material.PBRMaterial(
        vertexColors=True,
        doubleSided=True
    )
    trimesh_mesh.visual.material = material
    return trimesh_mesh


def concatenate_mesh_data(mesh_datas):
    n = 0
    vs, cs, fs = [], [], []
    for v, c, f in mesh_datas:
        vs.append(v)
        cs.append(c)
        fs.append(f + n)
        n += v.shape[0]
    return np.concatenate(vs, axis=0), np.concatenate(cs, axis=0), np.concatenate(fs, axis=0)


def gen_mesh_and_pcd(intr, depth, depth_valid, SE3=np.eye(4), rgb=None, valid_triangle=None, crop_region=None):
    '''
    :param intr: shape (4,)
    :param depth: shape (H, W)
    :param SE3: shape (4, 4), points coords: apply_SE3(SE3, depth_to_xyz(intr, depth))
    :param rgb:
        if rgb.dtype == np.uint8:
            use rgb / 255
        else:
            assert rgb.dtype == np.float32
            use rgb
    :param valid_triangle:
    :param crop_region: [lb_i, ub_i, lb_j, ub_j]
    :return:
    '''
    depth = depth.astype(np.float32)
    SE3 = SE3.astype(np.float32)

    H, W = depth.shape

    if crop_region is not None and len(crop_region) > 0:
        lb_i, ub_i, lb_j, ub_j = crop_region
        region_valid = np.zeros_like(depth_valid)
        region_valid[lb_i:ub_i, lb_j:ub_j] = True
        depth_valid = depth_valid & region_valid

    xyz = apply_SE3(SE3, depth_to_xyz(intr, depth))

    # create triangles
    triangle_v_idx = gen_triangle_v_idx(H, W)

    # compute validity based on xyz validity
    valid_flattened = depth_valid.reshape(-1)
    xyz_flattened = xyz.reshape(-1, 3)
    valid_triangle_vertex = \
        valid_flattened[triangle_v_idx[..., 0]] & \
        valid_flattened[triangle_v_idx[..., 1]] & \
        valid_flattened[triangle_v_idx[..., 2]]  # (H - 1, W - 1, 2)
    if valid_triangle is None:
        valid_triangle = valid_triangle_vertex
    else:
        valid_triangle = valid_triangle_vertex & valid_triangle

    if rgb is None:
        vertex_colors = .7 * np.ones_like(xyz_flattened)
    else:
        if rgb.dtype == np.uint8:
            vertex_colors = rgb.reshape(-1, 3).astype(np.float32) / 255.
        else:
            assert rgb.dtype == np.float32
            vertex_colors = rgb.reshape(-1, 3)

    pxl_displayed = np.zeros((H, W), dtype=np.bool_)
    pxl_displayed[:-1, :-1] |= valid_triangle[..., 0]
    pxl_displayed[1:, :-1] |= valid_triangle[..., 0]
    pxl_displayed[:-1, 1:] |= valid_triangle[..., 0]
    pxl_displayed[1:, 1:] |= valid_triangle[..., 1]
    pxl_displayed[1:, :-1] |= valid_triangle[..., 1]
    pxl_displayed[:-1, 1:] |= valid_triangle[..., 1]
    invisible_to_display = depth_valid & (~pxl_displayed)

    def get_up_xyz(depth):
        fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
        v, u = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing='ij')
        up_xyz = apply_SE3(SE3, np.stack([
            np.stack([((u - 1) - cx) / fx * depth, ((v - 1) - cy) / fy * depth, depth], axis=-1),
            np.stack([((u + 1) - cx) / fx * depth, ((v - 1) - cy) / fy * depth, depth], axis=-1),
            np.stack([((u - 1) - cx) / fx * depth, ((v + 1) - cy) / fy * depth, depth], axis=-1),
            np.stack([((u + 1) - cx) / fx * depth, ((v + 1) - cy) / fy * depth, depth], axis=-1),
        ], axis=-2).reshape(H, W, 2, 2, 3))
        return up_xyz

    depth_range = 1 / (.5 * (intr[0] + intr[1]))
    up_xyz_fnt = get_up_xyz((1 - depth_range) * depth)
    up_xyz_bck = get_up_xyz((1 + depth_range) * depth)

    up_xyz = np.stack([up_xyz_fnt, up_xyz_bck], axis=2).reshape(H, W, 8, 3)  # (H, W, 8, 3)
    up_vertex_idx = np.arange(H * W * 8).reshape(H, W, 8)
    up_triangles_to_stack = []
    for v1, v2, v3, v4 in [
        [0, 2, 3, 1],
        [0, 4, 6, 2],
        [0, 1, 5, 4],
        [7, 5, 1, 3],
        [7, 3, 2, 6],
        [7, 6, 4, 5],
    ]:
        up_triangles_to_stack.append(up_vertex_idx[..., [v1, v2, v3]])
        up_triangles_to_stack.append(up_vertex_idx[..., [v3, v4, v1]])
    up_triangles = np.stack(up_triangles_to_stack, axis=-2)  # (H, W, -1, 3)
    up_vertex_colors = np.repeat(vertex_colors.reshape(H, W, 1, 3), 8, axis=-2).reshape(-1, 3)

    xyz_flattened[~valid_flattened] = 0
    up_xyz[~depth_valid] = 0

    trimesh_mesh = gen_trimesh_mesh(*concatenate_mesh_data([
        (xyz_flattened, vertex_colors, triangle_v_idx[valid_triangle]),
        (up_xyz.reshape(-1, 3), up_vertex_colors, up_triangles[invisible_to_display].reshape(-1, 3))
    ]))

    pcd = gen_trimesh_mesh(up_xyz, up_vertex_colors, up_triangles[depth_valid].reshape(-1, 3))
    return trimesh_mesh, pcd
