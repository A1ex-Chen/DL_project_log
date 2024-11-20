#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

"""Matcher interface and Match class.

This module defines the Matcher interface and the Match object. The job of the
matcher is to match row and column indices based on the similarity matrix and
other optional parameters. Each column is matched to at most one row. There
are three possibilities for the matching:

1) match: A column matches a row.
2) no_match: A column does not match any row.
3) ignore: A column that is neither 'match' nor no_match.

The ignore case is regularly encountered in object detection: when an anchor has
a relatively small overlap with a ground-truth box, one neither wants to
consider this box a positive example (match) nor a negative example (no match).

The Match class is used to store the match results and it provides simple apis
to query the results.
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class Match(object):
  """Class to store results from the matcher.

  This class is used to store the results from the matcher. It provides
  convenient methods to query the matching results.
  """


  @property















class Matcher(object):
  """Abstract base class for matcher.
  """
  __metaclass__ = ABCMeta


  @abstractmethod
