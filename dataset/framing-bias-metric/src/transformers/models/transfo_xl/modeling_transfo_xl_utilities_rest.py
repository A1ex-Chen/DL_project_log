# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
 Utilities for PyTorch Transformer XL model. Directly adapted from https://github.com/kimiyoung/transformer-xl.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# CUDA_MAJOR = int(torch.version.cuda.split('.')[0])
# CUDA_MINOR = int(torch.version.cuda.split('.')[1])


class ProjectedAdaptiveLogSoftmax(nn.Module):


