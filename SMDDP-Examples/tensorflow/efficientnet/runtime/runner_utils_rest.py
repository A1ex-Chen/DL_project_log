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
import math

import tensorflow as tf
##
#import horovod.tensorflow as hvd
import smdistributed.dataparallel.tensorflow as sdp
##
from model import efficientnet_model

##
#from utils import dataset_factory, hvd_utils, callbacks, preprocessing
from utils import dataset_factory, sdp_utils, callbacks, preprocessing
##
__all__ = ['get_optimizer_params', 'get_metrics', 'get_learning_rate_params', 'build_model_params', 'get_models', 'build_augmenter_params', \
            'get_image_size_from_model', 'get_dataset_builders', 'build_stats', 'parse_inference_input', 'preprocess_image_files']














