#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Amazon Web Services. All rights reserved.
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

import sys
import logging
import os
import time
from datetime import datetime
import numpy as np
import collections
from sagemakercv.utils.runner import LogBuffer
from sagemakercv.utils.dist_utils import get_dist_info, master_only
from .priority import get_priority
from .hooks import Hook
import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops

class Runner(object):
    """A training helper.
    Args:
        model (:obj:`tf.keras.Model`): The model to be run.
        batch_processor (callable): A callable method that process a data
            batch. The interface of this method should be
            `batch_processor(model, data, train_mode) -> dict`
        optimizer (dict or :obj:`keras.Optimizer`): If it is a dict,
            runner will construct an optimizer according to it.
        work_dir (str, optional): The working directory to save checkpoints
            and logs.
        log_level (int): Logging level.
        logger (:obj:`logging.Logger`): Custom logger. If `None`, use the
            default logger.
    """
        
    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property

    @property
    
    @property
    
    @property
    
    @property
    
    @property
    
    @property
        
    
    @master_only

    
    

    
    