"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

""" Linear layer (alternate definition)
"""
import torch
import torch.nn.functional as F
from torch import nn as nn


class Linear(nn.Linear):