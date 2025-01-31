# Copyright (c) 2018-2019, NVIDIA CORPORATION
# Copyright (c) 2017-      Facebook, Inc
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from collections import OrderedDict
from numbers import Number
import dllogger
import numpy as np




PERF_METER = lambda: Meter(AverageMeter(), AverageMeter(), AverageMeter())
LOSS_METER = lambda: Meter(AverageMeter(), AverageMeter(), MinMeter())
ACC_METER = lambda: Meter(AverageMeter(), AverageMeter(), MaxMeter())
LR_METER = lambda: Meter(LastMeter(), LastMeter(), LastMeter())

LAT_100 = lambda: Meter(QuantileMeter(1), QuantileMeter(1), QuantileMeter(1))
LAT_99 = lambda: Meter(QuantileMeter(0.99), QuantileMeter(0.99), QuantileMeter(0.99))
LAT_95 = lambda: Meter(QuantileMeter(0.95), QuantileMeter(0.95), QuantileMeter(0.95))


class Meter(object):









class QuantileMeter(object):






class MaxMeter(object):






class MinMeter(object):






class LastMeter(object):






class AverageMeter(object):






class Logger(object):











