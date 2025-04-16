import json
from pathlib import Path

import PIL.Image
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Food101(VisionDataset):
    # Taken from https://github.com/pytorch/vision/blob/HEAD/torchvision/datasets/food101.py
    # Added for compatibility with old torchvision versions

    _URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    _MD5 = "85eeb15f3717b99a5da872d97d918f87"





