import numpy as np
import torch


def get_shifted_data(data, di, dj):
    H, W = data.shape
    shifted_data = data[max(di, 0): H + min(di, 0), max(dj, 0): W + min(dj, 0)]
    if isinstance(data, np.ndarray):
        if di < 0:
            shifted_data = np.concatenate([np.zeros_like(shifted_data[di:]), shifted_data], axis=0)
        if di > 0:
            shifted_data = np.concatenate([shifted_data, np.zeros_like(shifted_data[:di])], axis=0)
        if dj < 0:
            shifted_data = np.concatenate([np.zeros_like(shifted_data[:, dj:]), shifted_data], axis=1)
        if dj > 0:
            shifted_data = np.concatenate([shifted_data, np.zeros_like(shifted_data[:, :dj])], axis=1)
    elif isinstance(data, torch.Tensor):
        shifted_data = data[max(di, 0): H + min(di, 0), max(dj, 0): W + min(dj, 0)]
        if di < 0:
            shifted_data = torch.cat([torch.zeros_like(shifted_data[di:]), shifted_data], dim=0)
        if di > 0:
            shifted_data = torch.cat([shifted_data, torch.zeros_like(shifted_data[:di])], dim=0)
        if dj < 0:
            shifted_data = torch.cat([torch.zeros_like(shifted_data[:, dj:]), shifted_data], dim=1)
        if dj > 0:
            shifted_data = torch.cat([shifted_data, torch.zeros_like(shifted_data[:, :dj])], dim=1)
    return shifted_data
