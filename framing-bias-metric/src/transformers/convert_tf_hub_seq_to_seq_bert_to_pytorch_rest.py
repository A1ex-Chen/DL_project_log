# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Seq2Seq TF Hub checkpoint."""


import argparse

from transformers import (
    BertConfig,
    BertGenerationConfig,
    BertGenerationDecoder,
    BertGenerationEncoder,
    load_tf_weights_in_bert_generation,
    logging,
)


logging.set_verbosity_info()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_hub_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_encoder_named_decoder",
        action="store_true",
        help="If decoder has to be renamed to encoder in PyTorch model.",
    )
    parser.add_argument("--is_encoder", action="store_true", help="If model is an encoder.")
    parser.add_argument("--vocab_size", default=50358, type=int, help="Vocab size of model")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_hub_path,
        args.pytorch_dump_path,
        args.is_encoder_named_decoder,
        args.vocab_size,
        is_encoder=args.is_encoder,
    )