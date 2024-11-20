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
""" Auto Config class. """

import re
from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ..albert.configuration_albert import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, AlbertConfig
from ..bart.configuration_bart import BART_PRETRAINED_CONFIG_ARCHIVE_MAP, BartConfig
from ..bert.configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from ..bert_generation.configuration_bert_generation import BertGenerationConfig
from ..blenderbot.configuration_blenderbot import BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP, BlenderbotConfig
from ..camembert.configuration_camembert import CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, CamembertConfig
from ..ctrl.configuration_ctrl import CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP, CTRLConfig
from ..deberta.configuration_deberta import DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, DebertaConfig
from ..distilbert.configuration_distilbert import DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, DistilBertConfig
from ..dpr.configuration_dpr import DPR_PRETRAINED_CONFIG_ARCHIVE_MAP, DPRConfig
from ..electra.configuration_electra import ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
from ..encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from ..flaubert.configuration_flaubert import FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, FlaubertConfig
from ..fsmt.configuration_fsmt import FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP, FSMTConfig
from ..funnel.configuration_funnel import FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP, FunnelConfig
from ..gpt2.configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config
from ..layoutlm.configuration_layoutlm import LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP, LayoutLMConfig
from ..longformer.configuration_longformer import LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP, LongformerConfig
from ..lxmert.configuration_lxmert import LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP, LxmertConfig
from ..marian.configuration_marian import MarianConfig
from ..mbart.configuration_mbart import MBART_PRETRAINED_CONFIG_ARCHIVE_MAP, MBartConfig
from ..mobilebert.configuration_mobilebert import MobileBertConfig
from ..mt5.configuration_mt5 import MT5Config
from ..openai.configuration_openai import OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, OpenAIGPTConfig
from ..pegasus.configuration_pegasus import PegasusConfig
from ..prophetnet.configuration_prophetnet import PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ProphetNetConfig
from ..rag.configuration_rag import RagConfig
from ..reformer.configuration_reformer import ReformerConfig
from ..retribert.configuration_retribert import RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, RetriBertConfig
from ..roberta.configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig
from ..squeezebert.configuration_squeezebert import SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP, SqueezeBertConfig
from ..t5.configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config
from ..transfo_xl.configuration_transfo_xl import TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP, TransfoXLConfig
from ..xlm.configuration_xlm import XLM_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMConfig
from ..xlm_prophetnet.configuration_xlm_prophetnet import (
    XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    XLMProphetNetConfig,
)
from ..xlm_roberta.configuration_xlm_roberta import XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, XLMRobertaConfig
from ..xlnet.configuration_xlnet import XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, XLNetConfig


ALL_PRETRAINED_CONFIG_ARCHIVE_MAP = dict(
    (key, value)
    for pretrained_map in [
        # Add archive maps here
        BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BART_PRETRAINED_CONFIG_ARCHIVE_MAP,
        BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        MBART_PRETRAINED_CONFIG_ARCHIVE_MAP,
        OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        T5_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP,
        RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DPR_PRETRAINED_CONFIG_ARCHIVE_MAP,
        DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP,
        SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
        XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
        PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP,
    ]
    for key, value, in pretrained_map.items()
)


CONFIG_MAPPING = OrderedDict(
    [
        # Add configs here
        ("retribert", RetriBertConfig),
        ("mt5", MT5Config),
        ("t5", T5Config),
        ("mobilebert", MobileBertConfig),
        ("distilbert", DistilBertConfig),
        ("albert", AlbertConfig),
        ("bert-generation", BertGenerationConfig),
        ("camembert", CamembertConfig),
        ("xlm-roberta", XLMRobertaConfig),
        ("pegasus", PegasusConfig),
        ("marian", MarianConfig),
        ("mbart", MBartConfig),
        ("bart", BartConfig),
        ("blenderbot", BlenderbotConfig),
        ("reformer", ReformerConfig),
        ("longformer", LongformerConfig),
        ("roberta", RobertaConfig),
        ("deberta", DebertaConfig),
        ("flaubert", FlaubertConfig),
        ("fsmt", FSMTConfig),
        ("squeezebert", SqueezeBertConfig),
        ("bert", BertConfig),
        ("openai-gpt", OpenAIGPTConfig),
        ("gpt2", GPT2Config),
        ("transfo-xl", TransfoXLConfig),
        ("xlnet", XLNetConfig),
        ("xlm-prophetnet", XLMProphetNetConfig),
        ("prophetnet", ProphetNetConfig),
        ("xlm", XLMConfig),
        ("ctrl", CTRLConfig),
        ("electra", ElectraConfig),
        ("encoder-decoder", EncoderDecoderConfig),
        ("funnel", FunnelConfig),
        ("lxmert", LxmertConfig),
        ("dpr", DPRConfig),
        ("layoutlm", LayoutLMConfig),
        ("rag", RagConfig),
    ]
)

MODEL_NAMES_MAPPING = OrderedDict(
    [
        # Add full (and cased) model names here
        ("retribert", "RetriBERT"),
        ("t5", "T5"),
        ("mobilebert", "MobileBERT"),
        ("distilbert", "DistilBERT"),
        ("albert", "ALBERT"),
        ("bert-generation", "Bert Generation"),
        ("camembert", "CamemBERT"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("pegasus", "Pegasus"),
        ("blenderbot", "Blenderbot"),
        ("marian", "Marian"),
        ("mbart", "mBART"),
        ("bart", "BART"),
        ("reformer", "Reformer"),
        ("longformer", "Longformer"),
        ("roberta", "RoBERTa"),
        ("flaubert", "FlauBERT"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("squeezebert", "SqueezeBERT"),
        ("bert", "BERT"),
        ("openai-gpt", "OpenAI GPT"),
        ("gpt2", "OpenAI GPT-2"),
        ("transfo-xl", "Transformer-XL"),
        ("xlnet", "XLNet"),
        ("xlm", "XLM"),
        ("ctrl", "CTRL"),
        ("electra", "ELECTRA"),
        ("encoder-decoder", "Encoder decoder"),
        ("funnel", "Funnel Transformer"),
        ("lxmert", "LXMERT"),
        ("deberta", "DeBERTa"),
        ("layoutlm", "LayoutLM"),
        ("dpr", "DPR"),
        ("rag", "RAG"),
        ("xlm-prophetnet", "XLMProphetNet"),
        ("prophetnet", "ProphetNet"),
        ("mt5", "mT5"),
    ]
)





    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the :meth:`~transformers.AutoConfig.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """


    @classmethod

    @classmethod
    @replace_list_option_in_docstrings()