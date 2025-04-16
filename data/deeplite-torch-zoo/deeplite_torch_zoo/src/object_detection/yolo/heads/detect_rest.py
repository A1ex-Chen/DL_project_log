# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

import torch
import torch.nn as nn

from deeplite_torch_zoo.utils import no_jit_trace, check_version


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    no_post_processing = False  # don't export bbox decoding ops


