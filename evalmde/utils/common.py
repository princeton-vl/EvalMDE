from pathlib import Path
from datetime import datetime

import shortuuid
from omegaconf import DictConfig


def flatten_dict_cfg(cfg):  # [dict | DictConfig]) -> DictConfig:
    ret = {}
    if isinstance(cfg, dict):
        cfg = DictConfig(cfg)
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            ret_v = flatten_dict_cfg(v)
            for _k, _v in ret_v.items():
                ret[f'{k}_{_k}'] = _v
        else:
            ret[k] = v
    return DictConfig(ret)


def current_time():
    current_time = datetime.now()
    readable_time = current_time.strftime("%Y-%m-%d-%H:%M:%S")
    return readable_time


def uuid(length=8):
    """
    https://github.com/wandb/client/blob/master/wandb/util.py#L677
    """

    # ~3t run ids (36**8)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(length)


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of the filename:{file_name}')
    return file_name


def assign_item_to_dict(d: dict, ks: list, v):
    '''
    run d[ks[0]][ks[1]]...[ks[-1]] = v with filling empty keys
    :param d:
    :param ks:
    :param v:
    :return:
    '''
    k = ks[0]
    if len(ks) == 1:
        d[k] = v
    else:
        if k not in d:
            d[k] = dict()
        assign_item_to_dict(d[k], ks[1:], v)
