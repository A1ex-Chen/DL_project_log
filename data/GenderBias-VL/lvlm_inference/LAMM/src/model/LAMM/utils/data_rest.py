#!/usr/bin/env python3
# Portions Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import os
import math
import requests

import torch
import torch.nn as nn
# import torchaudio
import logging

from .multimodal_preprocessors import SimpleTokenizer
from PIL import Image
from pytorchvideo import transforms as pv_transforms
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

DEFAULT_AUDIO_FRAME_SHIFT_MS = 10  # in milliseconds

BPE_PATH = "../CLIP/bpe_simple_vocab_16e6.txt.gz"





