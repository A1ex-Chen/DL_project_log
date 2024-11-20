# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
from contextlib import contextmanager

import torch
import smdistributed.dataparallel.torch.distributed as dist

# def init_distributed(cuda):
#     """
#     Initializes distributed backend.

#     :param cuda: (bool) if True initializes nccl backend, if False initializes
#         gloo backend
#     """
#     world_size = int(os.environ.get('WORLD_SIZE', 1))
#     distributed = (world_size > 1)
#     if distributed:
#         backend = 'nccl' if cuda else 'gloo'
#         torch.distributed.init_process_group(backend=backend,
#                                              init_method='env://')
#         assert torch.distributed.is_initialized()
#     return distributed










@contextmanager