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

import tensorflow as tf
from sagemakercv.core import AnchorGenerator, ProposeROIs

class AnchorHead(tf.keras.Model):
    """Anchor-based head (RPN, RetinaNet, SSD, etc.).
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        feat_channels (int): Number of hidden channels. Used in child classes.
        anchor_generator (dict): Config dict for anchor generator
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
    """  # noqa: W605
        
    
    