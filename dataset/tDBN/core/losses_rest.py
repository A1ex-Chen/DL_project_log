# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Classification and regression loss functions for object detection.

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss

Classification losses:
 * WeightedSigmoidClassificationLoss
 * WeightedSoftmaxClassificationLoss
 * BootstrappedSigmoidClassificationLoss
"""
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torchplus


class Loss(object):
  """Abstract base class for loss functions."""
  __metaclass__ = ABCMeta

  def __call__(self,
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               scope=None,
               **params):
    """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
    if ignore_nan_targets:
      target_tensor = torch.where(torch.isnan(target_tensor),
                                prediction_tensor,
                                target_tensor)
    return self._compute_loss(prediction_tensor, target_tensor, **params)

  @abstractmethod
  def _compute_loss(self, prediction_tensor, target_tensor, **params):
    """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
    pass

class WeightedL2LocalizationLoss(Loss):
  """L2 localization loss function with anchorwise output support.

  Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2
  """
  def __init__(self, code_weights=None):
    super().__init__()
    if code_weights is not None:
      self._code_weights = np.array(code_weights, dtype=np.float32)
      self._code_weights = Variable(torch.from_numpy(self._code_weights))#XXX  remove .cuda())
    else:
      self._code_weights = None

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    diff = prediction_tensor - target_tensor
    if self._code_weights is not None:
      self._code_weights = self._code_weights.type_as(prediction_tensor)
      self._code_weights = self._code_weights.view(1, 1, -1)
      diff = self._code_weights * diff
    weighted_diff = diff * weights.unsqueeze(-1)
    square_diff = 0.5 * weighted_diff * weighted_diff
    return square_diff.sum(2)

class WeightedSmoothL1LocalizationLoss(Loss):
  """Smooth L1 localization loss function.

  The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
  otherwise, where x is the difference between predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """
  def __init__(self, sigma=3.0, code_weights=None, codewise=True):
    super().__init__()
    self._sigma = sigma
    if code_weights is not None:
      self._code_weights = np.array(code_weights, dtype=np.float32)
      self._code_weights = Variable(torch.from_numpy(self._code_weights)) ## XXX, remove .cuda
    else:
      self._code_weights = None
    self._codewise = codewise
  def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    diff = prediction_tensor - target_tensor
    if self._code_weights is not None:
      code_weights = self._code_weights.type_as(prediction_tensor)
      diff = code_weights.view(1, 1, -1) * diff
    abs_diff = torch.abs(diff)
    abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma**2)).type_as(abs_diff)
    loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
      + (abs_diff - 0.5 / (self._sigma**2)) * (1. - abs_diff_lt_1)
    if self._codewise:
      anchorwise_smooth_l1norm = loss
      if weights is not None:
        anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
    else:
      anchorwise_smooth_l1norm = torch.sum(loss, 2)#  * weights
      if weights is not None:
        anchorwise_smooth_l1norm *= weights
    return anchorwise_smooth_l1norm



  @abstractmethod

class WeightedL2LocalizationLoss(Loss):
  """L2 localization loss function with anchorwise output support.

  Loss[b,a] = .5 * ||weights[b,a] * (prediction[b,a,:] - target[b,a,:])||^2
  """


class WeightedSmoothL1LocalizationLoss(Loss):
  """Smooth L1 localization loss function.

  The smooth L1_loss is defined elementwise as .5 x^2 if |x|<1 and |x|-.5
  otherwise, where x is the difference between predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """

def _sigmoid_cross_entropy_with_logits(logits, labels):
  # to be compatible with tensorflow, we don't use ignore_idx
  loss = torch.clamp(logits, min=0) - logits * labels.type_as(logits)
  loss += torch.log1p(torch.exp(-torch.abs(logits)))
  # transpose_param = [0] + [param[-1]] + param[1:-1]
  # logits = logits.permute(*transpose_param)
  # loss_ftor = nn.NLLLoss(reduce=False)
  # loss = loss_ftor(F.logsigmoid(logits), labels)
  return loss

def _softmax_cross_entropy_with_logits(logits, labels):
  param = list(range(len(logits.shape)))
  transpose_param = [0] + [param[-1]] + param[1:-1]
  logits = logits.permute(*transpose_param) # [N, ..., C] -> [N, C, ...]
  loss_ftor = nn.CrossEntropyLoss(reduce=False)
  loss = loss_ftor(logits, labels.max(dim=-1)[1])
  return loss


class WeightedSigmoidClassificationLoss(Loss):
  """Sigmoid cross entropy classification loss function."""



class SigmoidFocalClassificationLoss(Loss):
  """Sigmoid focal cross entropy loss.

  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """




class SoftmaxFocalClassificationLoss(Loss):
  """Softmax focal cross entropy loss.

  Focal loss down-weights well classified examples and focusses on the hard
  examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.
  """




class WeightedSoftmaxClassificationLoss(Loss):
  """Softmax loss function."""




class BootstrappedSigmoidClassificationLoss(Loss):
  """Bootstrapped sigmoid cross entropy classification loss function.

  This loss uses a convex combination of training labels and the current model's
  predictions as training targets in the classification loss. The idea is that
  as the model improves over time, its predictions can be trusted more and we
  can use these predictions to mitigate the damage of noisy/incorrect labels,
  because incorrect labels are likely to be eventually highly inconsistent with
  other stimuli predicted to have the same label by the model.

  In "soft" bootstrapping, we use all predicted class probabilities, whereas in
  "hard" bootstrapping, we use the single class favored by the model.

  See also Training Deep Neural Networks On Noisy Labels with Bootstrapping by
  Reed et al. (ICLR 2015).
  """


