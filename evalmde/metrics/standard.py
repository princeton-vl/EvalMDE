# source: https://github.com/YvanYin/Metric3D/blob/main/mono/utils/avg_meter.py
import torch


def reformat_input(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    x = x.to(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    return x


def absrel_pnt(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 3 and target.dim() == 3 and mask.dim() == 2
    if mask.sum() == 0:
        return None, None

    dist_gt = torch.norm(target, dim=-1)
    dist_err = torch.norm(pred - target, dim=-1)
    err_heatmap = dist_err / (dist_gt + (1e-10)) * mask
    err_heatmap[mask < .5] = 0
    err = err_heatmap.sum() / mask.sum()
    return err_heatmap.cpu().numpy(), err.item()


def rel_depth(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, None
    mask = mask > .5
    p, t = pred[mask], target[mask]
    device = p.device
    N = p.shape[0]
    M = int(1e7)
    i = torch.randint(0, N, (M,), device=device, dtype=torch.long)
    j = torch.randint(0, N, (M,), device=device, dtype=torch.long)
    correct = (p[i] < p[j]) == (t[i] < t[j])
    return None, correct.float().mean().item()


def absrel(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, None

    t_m = target * mask
    p_m = pred * mask
    t_m[mask < .5] = 0
    p_m[mask < .5] = 0

    err_heatmap = torch.abs(t_m - p_m) / (t_m + 1e-10)  # (H, W)
    err = err_heatmap.sum() / mask.sum()
    return err_heatmap.cpu().numpy(), err.item()


def rmse(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, None

    t_m = target * mask
    p_m = pred * mask
    t_m[mask < .5] = 0
    p_m[mask < .5] = 0

    err_heatmap = (t_m - p_m) ** 2  # (H, W)
    err = torch.sqrt(err_heatmap.sum() / mask.sum())
    return err_heatmap.cpu().numpy(), err.item()


def rmse_log(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, None

    t_m = target * mask
    p_m = pred * mask
    t_m[mask < .5] = 0
    p_m[mask < .5] = 0

    err_heatmap = ((torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask) ** 2  # (H, W)
    err = torch.sqrt(err_heatmap.sum() / mask.sum())
    return err_heatmap.cpu().numpy(), err.item()


def delta1(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, (None, None, None)

    t_m = target * mask
    p_m = pred

    gt_pred = t_m / (p_m + 1e-10)  # (H, W)
    pred_gt = p_m / (t_m + 1e-10)  # (H, W)
    gt_pred_gt = torch.stack([gt_pred, pred_gt], dim=-1)  # (H, W, 2)
    ratio_max = torch.amax(gt_pred_gt, dim=-1)  # (H, W)
    err_heatmap = (ratio_max - 1) * mask  # (H, W)
    ratio_max[mask < .5] = 99999

    delta_1_sum = torch.sum(ratio_max < 1.25)
    delta_2_sum = torch.sum(ratio_max < 1.25 ** 2)
    delta_3_sum = torch.sum(ratio_max < 1.25 ** 3)
    return err_heatmap.cpu().numpy(), (delta_1_sum / mask.sum()).item()


def delta0125(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, (None, None, None)

    t_m = target * mask
    p_m = pred

    gt_pred = t_m / (p_m + 1e-10)  # (H, W)
    pred_gt = p_m / (t_m + 1e-10)  # (H, W)
    gt_pred_gt = torch.stack([gt_pred, pred_gt], dim=-1)  # (H, W, 2)
    ratio_max = torch.amax(gt_pred_gt, dim=-1)  # (H, W)
    err_heatmap = (ratio_max - 1) * mask  # (H, W)
    ratio_max[mask < .5] = 99999

    delta_sum = torch.sum(ratio_max < 1.25 ** 0.125)
    return err_heatmap.cpu().numpy(), (delta_sum / mask.sum()).item()


def delta2(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, (None, None, None)

    t_m = target * mask
    p_m = pred

    gt_pred = t_m / (p_m + 1e-10)  # (H, W)
    pred_gt = p_m / (t_m + 1e-10)  # (H, W)
    gt_pred_gt = torch.stack([gt_pred, pred_gt], dim=-1)  # (H, W, 2)
    ratio_max = torch.amax(gt_pred_gt, dim=-1)  # (H, W)
    err_heatmap = (ratio_max - 1) * mask  # (H, W)
    ratio_max[mask < .5] = 99999

    delta_1_sum = torch.sum(ratio_max < 1.25)
    delta_2_sum = torch.sum(ratio_max < 1.25 ** 2)
    delta_3_sum = torch.sum(ratio_max < 1.25 ** 3)
    return err_heatmap.cpu().numpy(), (delta_2_sum / mask.sum()).item()


def delta3(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, (None, None, None)

    t_m = target * mask
    p_m = pred

    gt_pred = t_m / (p_m + 1e-10)  # (H, W)
    pred_gt = p_m / (t_m + 1e-10)  # (H, W)
    gt_pred_gt = torch.stack([gt_pred, pred_gt], dim=-1)  # (H, W, 2)
    ratio_max = torch.amax(gt_pred_gt, dim=-1)  # (H, W)
    err_heatmap = (ratio_max - 1) * mask  # (H, W)
    ratio_max[mask < .5] = 99999

    delta_1_sum = torch.sum(ratio_max < 1.25)
    delta_2_sum = torch.sum(ratio_max < 1.25 ** 2)
    delta_3_sum = torch.sum(ratio_max < 1.25 ** 3)
    return err_heatmap.cpu().numpy(), (delta_3_sum / mask.sum()).item()


def log10(pred, target, mask):
    pred, target, mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert pred.dim() == 2 and target.dim() == 2 and mask.dim() == 2
    if mask.sum() == 0:
        return None, None

    t_m = target * mask
    p_m = pred * mask
    t_m[mask < .5] = 0
    p_m[mask < .5] = 0

    err_heatmap = torch.abs((torch.log10(p_m+1e-10) - torch.log10(t_m+1e-10)) * mask)
    err = err_heatmap.sum() / mask.sum()
    return err_heatmap.cpu().numpy(), err.item()


def rmse_log_si(pred, target, mask):  # RMSE (log, scale-invariant)
    # https://github.com/prs-eth/Marigold/blob/main/src/util/metric.py#L175
    depth_pred, depth_gt, valid_mask = reformat_input(pred), reformat_input(target), reformat_input(mask)
    assert depth_pred.dim() == 2 and depth_gt.dim() == 2 and valid_mask.dim() == 2
    if valid_mask.sum() == 0:
        return None, None

    valid_mask = valid_mask > .5
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term))
    return None, loss.item()
