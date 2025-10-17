import numpy as np


ROT_LIGHT_NUM_LIGHT = 30
ROT_LIGHT_NUM_LOOP = 3


def gen_rot_light__light_pos(num_light, num_loop):
    theta = np.linspace(0, np.pi, num_light)
    phi = np.linspace(0, 2 * np.pi * num_loop, num_light)
    x = np.sin(theta) * np.cos(phi)
    z = np.sin(theta) * np.sin(phi)
    y = np.cos(theta)
    return np.stack([x, y, z], axis=-1)
