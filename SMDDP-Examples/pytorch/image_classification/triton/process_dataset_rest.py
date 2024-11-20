#!/usr/bin/env python3
# Copyright (c) 2021 NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import tarfile
from pathlib import Path
from typing import Tuple, Dict, List

from PIL import Image
from tqdm import tqdm

DATASETS_DIR = os.environ.get("DATASETS_DIR", None)
IMAGENET_DIRNAME = "imagenet"
IMAGE_ARCHIVE_FILENAME = "ILSVRC2012_img_val.tar"
DEVKIT_ARCHIVE_FILENAME = "ILSVRC2012_devkit_t12.tar.gz"
LABELS_REL_PATH = "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
META_REL_PATH = "ILSVRC2012_devkit_t12/data/meta.mat"

TARGET_SIZE = (224, 224)  # (width, height)
_RESIZE_MIN = 256  # resize preserving aspect ratio to where this is minimal size








if __name__ == "__main__":
    main()