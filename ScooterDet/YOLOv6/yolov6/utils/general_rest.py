#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import glob
import math
import torch
import requests
import pkg_resources as pkg
from pathlib import Path
from yolov6.utils.events import LOGGER




















    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def download_ckpt(path):
    """Download checkpoints of the pretrained models"""
    basename = os.path.basename(path)
    dir = os.path.abspath(os.path.dirname(path))
    os.makedirs(dir, exist_ok=True)
    LOGGER.info(f"checkpoint {basename} not exist, try to downloaded it from github.")
    # need to update the link with every release
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.4.0/{basename}"
    LOGGER.warning(f"downloading url is: {url}, pealse make sure the version of the downloading model is correspoing to the code version!")
    r = requests.get(url, allow_redirects=True)
    assert r.status_code == 200, "Unable to download checkpoints, manually download it"
    open(path, 'wb').write(r.content)
    LOGGER.info(f"checkpoint {basename} downloaded and saved")


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        LOGGER.warning(f'--img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')
    return new_size


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check whether the package's version is match the required version.
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    if hard:
        info = f'⚠️ {name}{minimum} is required by YOLOv6, but {name}{current} is currently installed'
        assert result, info  # assert minimum version requirement
    return result