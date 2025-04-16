# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert RoBERTa checkpoint."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import numpy as np
import torch

from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from transformers.modeling_bert import (BertConfig, BertEncoder,
                                        BertIntermediate, BertLayer,
                                        BertModel, BertOutput,
                                        BertSelfAttention,
                                        BertSelfOutput)
from transformers.modeling_roberta import (RobertaEmbeddings,
                                           RobertaForMaskedLM,
                                           RobertaForSequenceClassification,
                                           RobertaModel)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_TEXT = 'Hello world! cécé herlolip'




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--roberta_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the official PyTorch dump.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    parser.add_argument("--classification_head",
                        action = "store_true",
                        help = "Whether to convert a final classification head.")
    args = parser.parse_args()
    convert_roberta_checkpoint_to_pytorch(
        args.roberta_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.classification_head
    )
