# https://github.com/meituan/YOLOv6/

import numpy as np
import torch
import torch.nn.functional as F












#   This has not been tested with non-square kernels (kernel.size(2) != kernel.size(3)) nor even-size kernels