from pathlib import Path

import PIL.Image
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
    verify_str_arg,
)
from torchvision.datasets.vision import VisionDataset


class Flowers102(VisionDataset):
    # Taken from https://github.com/pytorch/vision/blob/HEAD/torchvision/datasets/flowers102.py
    # Added for compatibility with old torchvision versions

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}





