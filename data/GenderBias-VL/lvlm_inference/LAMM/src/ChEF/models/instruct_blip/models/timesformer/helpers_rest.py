"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Based on https://github.com/facebookresearch/TimeSformer
"""

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified model creation / weight loading / state_dict helpers

import logging, warnings
import os
import math
from collections import OrderedDict

import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F






# def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
#     resume_epoch = None
# if os.path.isfile(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location='cpu')
#     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#         if log_info:
#             _logger.info('Restoring model state from checkpoint...')
#         new_state_dict = OrderedDict()
#         for k, v in checkpoint['state_dict'].items():
#             name = k[7:] if k.startswith('module') else k
#             new_state_dict[name] = v
#         model.load_state_dict(new_state_dict)

#         if optimizer is not None and 'optimizer' in checkpoint:
#             if log_info:
#                 _logger.info('Restoring optimizer state from checkpoint...')
#             optimizer.load_state_dict(checkpoint['optimizer'])

#         if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
#             if log_info:
#                 _logger.info('Restoring AMP loss scaler state from checkpoint...')
#             loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

#         if 'epoch' in checkpoint:
#             resume_epoch = checkpoint['epoch']
#             if 'version' in checkpoint and checkpoint['version'] > 1:
#                 resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

#         if log_info:
#             _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
#     else:
#         model.load_state_dict(checkpoint)
#         if log_info:
#             _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
#     return resume_epoch
# else:
#     _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
#     raise FileNotFoundError()













