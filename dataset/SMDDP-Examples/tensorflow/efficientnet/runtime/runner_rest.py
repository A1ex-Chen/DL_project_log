# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import print_function

import os
import multiprocessing
import warnings
import yaml
import time

import tensorflow as tf
import numpy as np

##
#import horovod.tensorflow.keras as hvd
import smdistributed.dataparallel.tensorflow as sdp
from smdistributed.dataparallel.tensorflow.keras import callbacks as sdp_callbacks
import smdistributed.dataparallel.tensorflow.keras as sdp_keras
##

##
#from utils import hvd_utils, optimizer_factory
from utils import sdp_utils, optimizer_factory
##
from utils import callbacks as custom_callbacks

from runtime.runner_utils import get_optimizer_params, get_metrics, get_learning_rate_params, \
                        build_model_params, get_models, get_dataset_builders, build_stats, \
                        parse_inference_input, preprocess_image_files

__all__ = [
    'Runner',
]

DTYPE_MAP = {
    'float32': tf.float32,
    'bfloat16': tf.bfloat16,
    'float16': tf.float16,
    'fp32': tf.float32,
    'bf16': tf.bfloat16,
}

class Runner(object):












