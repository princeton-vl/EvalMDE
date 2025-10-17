from typing import List

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def get_angle_between(n1: torch.Tensor, n2: torch.Tensor) -> torch.Tensor:
    '''
    :param n1: shape (..., 3), norm > 0
    :param n2: shape (..., 3), norm > 0
    :return: shape (...), in radius
    '''
    return torch.acos((F.normalize(n1, dim=-1) * F.normalize(n2, dim=-1)).sum(dim=-1).clamp(-1, 1))


def reformat_as_torch_tensor(x, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
    if isinstance(x, List):
        return torch.tensor(x, device=device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device)
    elif isinstance(x, torch.Tensor):
        return x.to(device=device)
    else:
        raise ValueError(f'Unsupported type: {type(x)}')
