import os
from pathlib import Path

import PIL.Image
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision.datasets.vision import VisionDataset


IMAGEWOOF_IMAGENET_CLS_LABEL_MAP = (155, 159, 162, 167, 182, 193, 207, 229, 258, 273)


class Imagewoof(VisionDataset):




