# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from contextlib import contextmanager
from functools import wraps
import torch

__all__ = ["retry_if_cuda_oom"]


@contextmanager



    @wraps(func)

    return wrapped