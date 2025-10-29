import os.path as osp
from setuptools import setup, find_packages

ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='evalmde',
    packages=find_packages(),
    install_requires=[
        "numpy==2.0.0",
        "opencv-python==4.12.0.88",
        "open3d==0.19.0",
        "pyglet==1.5.28",
        "imageio==2.33.1",
        "hydra-core==1.3.0",
        "pyrender==0.1.45",
        "evo==1.26.0",
        "loguru==0.7.2",
        "shortuuid==1.0.13",
        "DateTime==5.5",
        "plyfile==1.1",
        "HTML4Vision==0.4.3",
        "timm==1.0.9",
        "imgaug==0.4.0",
        "iopath==0.1.10",
        "imagecorruptions==1.1.2",
        "mmcv==2.2.0",
        "gitpython==3.1.44",
        "pomegranate==1.1.1",
        "matplotlib==3.9.0",
        "wandb==0.22.2",
        "cvxpy==1.6.5",
        "mathutils==3.3.0",
        "OpenEXR==3.3.3",
        "Imath==0.0.2",
        "pywavelets==1.8.0",
        "h5py==3.14.0",
    ],
)

