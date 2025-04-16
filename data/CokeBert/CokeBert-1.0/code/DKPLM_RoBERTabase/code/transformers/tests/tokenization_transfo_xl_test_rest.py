# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
import pytest
from io import open

from transformers import is_torch_available

if is_torch_available():
    import torch
    from transformers.tokenization_transfo_xl import TransfoXLTokenizer, VOCAB_FILES_NAMES
else:
    pytestmark = pytest.mark.skip("Require Torch")  # TODO: untangle Transfo-XL tokenizer from torch.load and torch.save

from .tokenization_tests_commons import CommonTestCases

class TransfoXLTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = TransfoXLTokenizer if is_torch_available() else None








if __name__ == '__main__':
    unittest.main()