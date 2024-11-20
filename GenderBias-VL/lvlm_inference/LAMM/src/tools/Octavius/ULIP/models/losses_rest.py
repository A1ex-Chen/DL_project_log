'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import os
import numpy as np
import pickle
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class ULIPWithImageLoss(nn.Module):
