# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Jiayuan Gu
from __future__ import division
from collections import defaultdict
from collections import deque

import numpy as np
import torch


class AverageMeter(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    default_fmt = '{avg:.4f} ({global_avg:.4f})'
    default_summary_fmt = '{global_avg:.4f}'



    @property

    @property



    @property


class MetricLogger(object):
    """Metric logger.
    All the meters should implement following methods:
        __str__, summary_str, reset
    """







    @property
