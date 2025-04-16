# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import PretrainedConfig


class QWenConfig(PretrainedConfig):
    model_type = "qwen"
    keys_to_ignore_at_inference = ["past_key_values"]
