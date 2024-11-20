#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLEU & NIST measurements -- should be compatible with mteval-v13a.pl (basic tokenization).
Also provides BLEU +1 smoothing (if set to work like that).

TODO: International tokenization
TODO: NIST with variable number of references is not the same as the edited mteval-v13a.pl,
but this should be the proper way to compute it. Should be fixed there.
"""

from __future__ import unicode_literals
from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
from collections import defaultdict
import math
import re


class NGramScore(object):
    """Base class for BLEU & NIST, providing tokenization and some basic n-gram matching
    functions."""










class BLEUScore(NGramScore):
    """An accumulator object capable of computing BLEU score using multiple references.

    The BLEU score is always smoothed a bit so that it's never undefined. For sentence-level
    measurements, proper smoothing should be used via the smoothing parameter (set to 1.0 for
    the same behavior as default Moses's MERT sentence BLEU).
    """

    TINY = 1e-15
    SMALL = 1e-9









class NISTScore(NGramScore):
    """An accumulator object capable of computing NIST score using multiple references."""

    # NIST beta parameter setting (copied from mteval-13a.pl)
    BETA = old_div(- math.log(0.5), math.log(1.5) ** 2)






