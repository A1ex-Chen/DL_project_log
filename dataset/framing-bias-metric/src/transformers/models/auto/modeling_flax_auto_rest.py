# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
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
""" Auto Model class. """


from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..bert.modeling_flax_bert import FlaxBertModel
from ..roberta.modeling_flax_roberta import FlaxRobertaModel
from .configuration_auto import AutoConfig, BertConfig, RobertaConfig


logger = logging.get_logger(__name__)


ALL_PRETRAINED_MODEL_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        FlaxBertModel.pretrained_model_archive_map,
        FlaxRobertaModel.pretrained_model_archive_map,
    ]
    for key, value, in pretrained_map.items()
)

FLAX_MODEL_MAPPING = OrderedDict(
    [
        (RobertaConfig, FlaxRobertaModel),
        (BertConfig, FlaxBertModel),
    ]
)


class FlaxAutoModel(object):
    r"""
    :class:`~transformers.FlaxAutoModel` is a generic model class that will be instantiated as one of the base model
    classes of the library when created with the `FlaxAutoModel.from_pretrained(pretrained_model_name_or_path)` or the
    `FlaxAutoModel.from_config(config)` class methods.

    This class cannot be instantiated using `__init__()` (throws an error).
    """


    @classmethod

    @classmethod