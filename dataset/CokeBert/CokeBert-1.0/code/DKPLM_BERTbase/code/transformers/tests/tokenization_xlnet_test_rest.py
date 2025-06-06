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

from transformers.tokenization_xlnet import (XLNetTokenizer, SPIECE_UNDERLINE)

from .tokenization_tests_commons import CommonTestCases

SAMPLE_VOCAB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'fixtures/test_sentencepiece.model')

class XLNetTokenizationTest(CommonTestCases.CommonTokenizerTester):

    tokenizer_class = XLNetTokenizer








    @pytest.mark.slow


if __name__ == '__main__':
    unittest.main()