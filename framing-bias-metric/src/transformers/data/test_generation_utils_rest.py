import random
import unittest

import timeout_decorator

from transformers import is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_torch


if is_torch_available():
    import torch

    from transformers import MarianConfig, MarianMTModel


@require_torch
class GenerationUtilsTest(unittest.TestCase):
    @cached_property

    @cached_property


    @timeout_decorator.timeout(10)