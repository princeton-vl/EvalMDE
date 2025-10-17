import argparse
import math
from pathlib import Path
import json

from PIL import Image
import cv2
import torch
from torchvision import transforms as torch_trans
import numpy as np

from evalmde.utils.proj import depth_to_xyz
from evalmde.utils.common import assign_item_to_dict, pathlib_file
from evalmde.utils.image import resize
from evalmde.utils.image import imread_rgb
from evalmde.utils.depth import load_data
from evalmde.utils.np_and_th import get_shifted_data


@torch.no_grad()
def compute_grid_lb_ub(data, i, j):
    '''
    .           .           .

          -------------
          |(0,0)|(0,1)|
    .     ------.------     .
          |(1,0)|(1,1)|
          -------------

    .           .           .
    '''
    if i == 0 and j == 0:
        x00 = .25 * (data[:-1, :-1] + data[:-1, 1:] + data[1:, :-1] + data[1:, 1:])
        x01 = .5 * (data[:-1, 1:] + data[1:, 1:])
        x10 = .5 * (data[1:, :-1] + data[1:, 1:])
        x11 = 1. * data[1:, 1:]
    elif i == 0 and j == 1:
        x00 = .5 * (data[:-1, :-1] + data[1:, :-1])
        x01 = .25 * (data[:-1, :-1] + data[:-1, 1:] + data[1:, :-1] + data[1:, 1:])
        x10 = 1. * data[1:, :-1]
        x11 = .5 * (data[1:, :-1] + data[1:, 1:])
    elif i == 1 and j == 0:
        x00 = .5 * (data[:-1, :-1] + data[:-1, 1:])
        x01 = 1. * data[:-1, 1:]
        x10 = .25 * (data[:-1, :-1] + data[:-1, 1:] + data[1:, :-1] + data[1:, 1:])
        x11 = .5 * (data[:-1, 1:] + data[1:, 1:])
    else:
        x00 = 1. * data[:-1, :-1]
        x01 = .5 * (data[:-1, :-1] + data[:-1, 1:])
        x10 = .5 * (data[:-1, :-1] + data[1:, :-1])
        x11 = .25 * (data[:-1, :-1] + data[:-1, 1:] + data[1:, :-1] + data[1:, 1:])
    x = torch.stack([x00, x01, x10, x11], dim=-1)
    lb, ub = x.min(dim=-1).values, x.max(dim=-1).values  # (H - 1, W - 1), (H - 1, W - 1)
    del x
    return lb, ub


@torch.no_grad()
def compute_high_res_idx(high_res_shape, data_low_res, valid, valid_high_res, gap, val_lb):
    '''
    :param high_res_shape: (Hu, Wu)
    :param data_low_res: shape (Hl, Wl)
    :param valid_high_res: shape (Hl, Wl)
    :param gap:
    :param val_lb:
    :return: res_high_res
        res_high_res: shape (Hu, Wu)
    '''
    Hu, Wu = high_res_shape

    # fill invalid pixels with neighbor means
    data_low_res = data_low_res.clone()
    nb_data_sum = torch.zeros_like(data_low_res)
    nb_data_cnt = torch.zeros_like(data_low_res)
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            nb_valid = get_shifted_data(valid, di, dj)
            nb_data = get_shifted_data(data_low_res, di, dj)
            nb_data_sum[nb_valid] += nb_data[nb_valid]
            nb_data_cnt[nb_valid] += 1
    nb_data_sum[nb_data_cnt < .5] = 0
    data_low_res[~valid] = (nb_data_sum / nb_data_cnt.clamp(min=1))[~valid]

    data_high_res = torch_trans.functional.resize(data_low_res[None], (Hu, Wu), torch_trans.InterpolationMode.BILINEAR)[0]
    res_high_res = -torch.ones((Hu, Wu), dtype=torch.int32, device=data_high_res.device)

    for i in range(2):
        for j in range(2):
            lb, ub = compute_grid_lb_ub(data_high_res, i, j)
            lb_i = torch.clip(torch.ceil((lb - val_lb) / gap), min=0).to(res_high_res.dtype)
            ub_i = torch.clip(torch.floor((ub - val_lb) / gap), max=2e9).to(res_high_res.dtype)

            multi_line_mask = (lb_i < ub_i) | ((lb_i == ub_i) & (res_high_res[1 - i: Hu - i, 1 - j: Wu - j] != -1))
            single_line_mask = (lb_i == ub_i) & (res_high_res[1 - i: Hu - i, 1 - j: Wu - j] == -1)

            res_high_res[1 - i: Hu - i, 1 - j: Wu - j][single_line_mask] = lb_i[single_line_mask]

            multi_line_upd_idx = torch.clip(torch.round((data_high_res[1 - i: Hu - i, 1 - j: Wu - j] - val_lb) / gap), min=0, max=2e9).to(res_high_res.dtype)
            multi_line_upd_idx = torch.where(multi_line_upd_idx < lb_i, lb_i, multi_line_upd_idx)
            multi_line_upd_idx = torch.where(multi_line_upd_idx > ub_i, ub_i, multi_line_upd_idx)
            multi_line_upd_mask = ((res_high_res[1 - i: Hu - i, 1 - j: Wu - j] == -1) | (
                torch.abs(data_high_res[1 - i: Hu - i, 1 - j: Wu - j] - (res_high_res[1 - i: Hu - i, 1 - j: Wu - j] * gap + val_lb)) >
                torch.abs(data_high_res[1 - i: Hu - i, 1 - j: Wu - j] - (multi_line_upd_idx * gap + val_lb))
            )) & multi_line_mask
            res_high_res[1 - i: Hu - i, 1 - j: Wu - j][multi_line_upd_mask] = multi_line_upd_idx[multi_line_upd_mask]
    res_high_res[~valid_high_res] = -1
    return res_high_res


def get_contour_line_gap(data: torch.Tensor, valid: torch.Tensor, num_gap, qt):
    if not valid.any():
        return 1
    qt_lb = data[valid].quantile(qt).item()
    qt_ub = data[valid].quantile(1 - qt).item()
    gap = (qt_ub - qt_lb) / (num_gap * (1 - qt * 2))
    return gap


@torch.no_grad()
def gen_contour_line(rgb_high_res, data, valid, valid_high_res, is_z, num_gap, shift, thickness=0, qt=0.05, colormap=cv2.COLORMAP_JET):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.from_numpy(data).to(device)
    valid = torch.from_numpy(valid).to(device)
    valid_high_res = torch.from_numpy(valid_high_res).to(device)
    if is_z:
        data = 1 / data

    gap = get_contour_line_gap(data, valid, num_gap, qt)
    data_lb = data[valid].min().item() if valid.any() else 0
    val_lb = data_lb + gap * shift

    high_res_shape = rgb_high_res.shape[:2]
    res_high_res = compute_high_res_idx(high_res_shape, data, valid, valid_high_res, gap, val_lb)

    res = res_high_res.clone()
    dlt_rng = int(math.floor(thickness))
    for di in range(-dlt_rng, dlt_rng + 1):
        for dj in range(-dlt_rng, dlt_rng + 1):
            if di * di + dj * dj > thickness * thickness:
                continue
            nb_res = get_shifted_data(res_high_res, di, dj)
            upd_mask = get_shifted_data(valid_high_res, di, dj) & (res == -1) & (nb_res != -1) & valid_high_res
            res[upd_mask] = nb_res[upd_mask]

    if (res != -1).any():
        res[res != -1] -= res[res != -1].min()
    res = res.cpu().numpy()
    num_val = max(2, res.max().item() + 1)
    valid_high_res = valid_high_res.cpu().numpy()

    base_col = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[None], colormap)[0].astype(np.float32)  # (256, 3)
    idx = np.arange(num_val, dtype=np.float32) / (num_val - 1) * 255  # (itr,)
    idx_lb = np.floor(idx).astype(np.int32)  # (itr,)
    coef_lb = (idx_lb.astype(np.float32) + 1 - idx)[:, None]  # (itr, 1)
    col = base_col[idx_lb] * coef_lb + base_col[np.clip(idx_lb + 1, a_min=None, a_max=255)] * (1 - coef_lb)  # (itr, 3)
    col = np.round(col).astype(np.uint8)

    img = np.zeros_like(rgb_high_res)
    non_colored_mask = valid_high_res & (res == -1)
    img[non_colored_mask] = rgb_high_res[non_colored_mask]
    colored_mask = valid_high_res & (res != -1)
    img[colored_mask] = col[res[colored_mask]]
    return img, colored_mask, col


def pil_ds(img: np.ndarray, H, W):
    pil_img = Image.fromarray(img, mode='RGB')
    pil_img = pil_img.resize((W, H), Image.Resampling.LANCZOS)
    return np.array(pil_img)


def render_contour_line_imgs(xyz: np.ndarray, valid: np.ndarray, rgb_low_res: np.ndarray, save_shape, out_root):
    '''
    :param xyz:
    :param valid:
    :param rgb_low_res:
    :param save_shape: (H, W)
    :param out_root:
    :return:
    '''
    # hyperparams
    texture_strength = 0.8
    draw_dim_lb = 4 * np.linalg.norm([1920, 1080])

    out_root = pathlib_file(out_root)

    dim = np.linalg.norm(rgb_low_res.shape[:2])
    us_sc = int(math.ceil(draw_dim_lb / dim))
    us_shape = (us_sc * rgb_low_res.shape[0], us_sc * rgb_low_res.shape[1])
    rgb_high_res = np.round(texture_strength * cv2.resize(rgb_low_res, (us_shape[1], us_shape[0]))).astype(np.uint8)
    valid_high_res = torch_trans.functional.resize(torch.from_numpy(valid)[None], rgb_high_res.shape[:2], torch_trans.InterpolationMode.NEAREST_EXACT)[0].numpy()

    summary = {}
    for thickness in [5 * np.linalg.norm(rgb_high_res.shape[:2]) / (4 * np.linalg.norm([1920, 1080]))]:
        for rel_num_gap in [0.015, 0.03, 0.06, 0.09, 0.12, 0.24, 0.42, 0.6]:
            num_gap = int(dim * rel_num_gap)
            for shift in [0.5]:
                imgs, colored_masks, col_maps = {}, {}, {}
                for i, name in enumerate(['x', 'y', 'z']):
                    imgs[name], colored_masks[name], col_maps[name] = \
                        gen_contour_line(rgb_high_res, xyz[..., i], valid, valid_high_res, name == 'z',
                                         num_gap, shift, thickness)

                    out_f = out_root / name / f'thickness__{thickness:.1f}___num_gap__{num_gap}___shift__{shift:.2f}.png'
                    out_f.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(out_f.as_posix(), pil_ds(imgs[name][us_sc:-us_sc, us_sc:-us_sc, ::-1].copy(), save_shape[0], save_shape[1]))
                    print(f'Saved to {out_f.resolve()}')
                    assign_item_to_dict(summary, [name, thickness, num_gap, shift], str(out_f.resolve().relative_to(out_root.resolve())))

                img_xy = rgb_high_res.copy()
                img_xy[np.logical_and(colored_masks['x'], colored_masks['y'])] = np.round(.5 * (imgs['x'].astype(np.float32) + imgs['y'].astype(np.float32))).astype(np.uint8)[np.logical_and(colored_masks['x'], colored_masks['y'])]
                img_xy[np.logical_and(colored_masks['x'], np.logical_not(colored_masks['y']))] = imgs['x'][np.logical_and(colored_masks['x'], np.logical_not(colored_masks['y']))]
                img_xy[np.logical_and(np.logical_not(colored_masks['x']), colored_masks['y'])] = imgs['y'][np.logical_and(np.logical_not(colored_masks['x']), colored_masks['y'])]
                img_xy[~valid_high_res] = 0
                # img_xy = caption_img_xy(img_xy, col_maps)
                out_f = out_root / 'xy' / f'thickness__{thickness:.1f}___num_gap__{num_gap}___shift__{shift:.2f}.png'
                out_f.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(out_f.as_posix(), pil_ds(img_xy[us_sc:-us_sc, us_sc:-us_sc, ::-1].copy(), save_shape[0], save_shape[1]))
                print(f'Saved to {out_f.resolve()}')
                assign_item_to_dict(summary, ['xy', thickness, num_gap, shift], str(out_f.resolve().relative_to(out_root.resolve())))
    with (out_root / 'summary.json').open('w') as F:
        json.dump(summary, F)


def get_out_dir(work_dir, depth_f):
    return work_dir / 'contour_line' / str((work_dir / depth_f).resolve().relative_to(work_dir.resolve()))[:-4].replace('/', '_')


def main(args):
    save_dim_ub = args.save_dim_ub

    root = args.root
    rgb_f = root / args.rgb_f
    data_f = root / args.depth_f

    raw_rgb = imread_rgb(rgb_f)

    save_sc = int(math.floor(save_dim_ub / np.linalg.norm(raw_rgb.shape[:2])))
    save_shape = (save_sc * raw_rgb.shape[0], save_sc * raw_rgb.shape[1])

    depth, intr, valid = load_data(data_f)
    xyz = depth_to_xyz(intr, depth)

    render_contour_line_imgs(xyz, valid, raw_rgb, save_shape, get_out_dir(root, args.depth_f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--depth_f", type=str, help='Path to depth file, relative to root.')
    parser.add_argument('--rgb_f', type=str, nargs='?', const=None, default='rgb.png', help='Path to rgb file, relative to root.')
    parser.add_argument("--save_dim_ub", type=float, default=np.linalg.norm([1920, 1080]))
    args = parser.parse_args()

    main(args)
