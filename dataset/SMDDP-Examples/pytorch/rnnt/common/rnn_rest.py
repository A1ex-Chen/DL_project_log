# Copyright (c) 2019, Myrtle Software Limited. All rights reserved.
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
from torch.nn import Parameter
from mlperf import logging




class LSTM(torch.nn.Module):





class DecoupledBase(torch.nn.Module):
    """Base class for decoupled RNNs.

    Meant for being sub-classed, with children class filling self.rnn
    with RNN cells.
    """






class DecoupledLSTM(DecoupledBase):