#
#  QAT_quantizer.py
#  YOLOv6
#
#  Created by Meituan on 2022/06/24.
#  Copyright © 2022
#

from absl import logging
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules

# Call this function before defining the model

# def QAT_quantizer():
# coming soon