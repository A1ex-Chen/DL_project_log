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

from transformers import is_tf_available

if is_tf_available():
    from transformers import (AutoConfig, BertConfig,
                                      TFAutoModel, TFBertModel,
                                      TFAutoModelWithLMHead, TFBertForMaskedLM,
                                      TFAutoModelForSequenceClassification, TFBertForSequenceClassification,
                                      TFAutoModelForQuestionAnswering, TFBertForQuestionAnswering)
    from transformers.modeling_tf_bert import TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP

    from .modeling_common_test import (CommonTestCases, ids_tensor)
    from .configuration_common_test import ConfigTester
else:
    pytestmark = pytest.mark.skip("Require TensorFlow")


class TFAutoModelTest(unittest.TestCase):





if __name__ == "__main__":
    unittest.main()