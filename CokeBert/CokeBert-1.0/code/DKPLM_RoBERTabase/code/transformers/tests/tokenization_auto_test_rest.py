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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import shutil
import pytest
import logging

from transformers import AutoTokenizer, BertTokenizer, AutoTokenizer, GPT2Tokenizer
from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP


class AutoTokenizerTest(unittest.TestCase):
    @pytest.mark.slow


if __name__ == "__main__":
    unittest.main()