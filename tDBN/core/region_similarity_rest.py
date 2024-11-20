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

"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from abc import ABCMeta
from abc import abstractmethod

from tDBN.core import box_np_ops

class RegionSimilarityCalculator(object):
  """Abstract base class for 2d region similarity calculator."""
  __metaclass__ = ABCMeta


  @abstractmethod


class RotateIouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Union (IOU) metric.

  This class computes pairwise similarity between two BoxLists based on IOU.
  """



class NearestIouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on the squared distance metric.

  This class computes pairwise similarity between two BoxLists based on the
  negative squared distance metric.
  """



class DistanceSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Area (IOA) metric.

  This class computes pairwise similarity between two BoxLists based on their
  pairwise intersections divided by the areas of tDBN BoxLists.
  """
