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
""" Auto Tokenizer class. """


from collections import OrderedDict

from ...configuration_utils import PretrainedConfig
from ...file_utils import is_sentencepiece_available, is_tokenizers_available
from ...utils import logging
from ..bart.tokenization_bart import BartTokenizer
from ..bert.tokenization_bert import BertTokenizer
from ..bert_japanese.tokenization_bert_japanese import BertJapaneseTokenizer
from ..bertweet.tokenization_bertweet import BertweetTokenizer
from ..blenderbot.tokenization_blenderbot import BlenderbotSmallTokenizer
from ..ctrl.tokenization_ctrl import CTRLTokenizer
from ..deberta.tokenization_deberta import DebertaTokenizer
from ..distilbert.tokenization_distilbert import DistilBertTokenizer
from ..dpr.tokenization_dpr import DPRQuestionEncoderTokenizer
from ..electra.tokenization_electra import ElectraTokenizer
from ..flaubert.tokenization_flaubert import FlaubertTokenizer
from ..fsmt.tokenization_fsmt import FSMTTokenizer
from ..funnel.tokenization_funnel import FunnelTokenizer
from ..gpt2.tokenization_gpt2 import GPT2Tokenizer
from ..herbert.tokenization_herbert import HerbertTokenizer
from ..layoutlm.tokenization_layoutlm import LayoutLMTokenizer
from ..longformer.tokenization_longformer import LongformerTokenizer
from ..lxmert.tokenization_lxmert import LxmertTokenizer
from ..mobilebert.tokenization_mobilebert import MobileBertTokenizer
from ..openai.tokenization_openai import OpenAIGPTTokenizer
from ..phobert.tokenization_phobert import PhobertTokenizer
from ..prophetnet.tokenization_prophetnet import ProphetNetTokenizer
from ..rag.tokenization_rag import RagTokenizer
from ..retribert.tokenization_retribert import RetriBertTokenizer
from ..roberta.tokenization_roberta import RobertaTokenizer
from ..squeezebert.tokenization_squeezebert import SqueezeBertTokenizer
from ..transfo_xl.tokenization_transfo_xl import TransfoXLTokenizer
from ..xlm.tokenization_xlm import XLMTokenizer
from .configuration_auto import (
    AlbertConfig,
    AutoConfig,
    BartConfig,
    BertConfig,
    BertGenerationConfig,
    BlenderbotConfig,
    CamembertConfig,
    CTRLConfig,
    DebertaConfig,
    DistilBertConfig,
    DPRConfig,
    ElectraConfig,
    EncoderDecoderConfig,
    FlaubertConfig,
    FSMTConfig,
    FunnelConfig,
    GPT2Config,
    LayoutLMConfig,
    LongformerConfig,
    LxmertConfig,
    MarianConfig,
    MBartConfig,
    MobileBertConfig,
    MT5Config,
    OpenAIGPTConfig,
    PegasusConfig,
    ProphetNetConfig,
    RagConfig,
    ReformerConfig,
    RetriBertConfig,
    RobertaConfig,
    SqueezeBertConfig,
    T5Config,
    TransfoXLConfig,
    XLMConfig,
    XLMProphetNetConfig,
    XLMRobertaConfig,
    XLNetConfig,
    replace_list_option_in_docstrings,
)


if is_sentencepiece_available():
    from ..albert.tokenization_albert import AlbertTokenizer
    from ..barthez.tokenization_barthez import BarthezTokenizer
    from ..bert_generation.tokenization_bert_generation import BertGenerationTokenizer
    from ..camembert.tokenization_camembert import CamembertTokenizer
    from ..marian.tokenization_marian import MarianTokenizer
    from ..mbart.tokenization_mbart import MBartTokenizer
    from ..mt5 import MT5Tokenizer
    from ..pegasus.tokenization_pegasus import PegasusTokenizer
    from ..reformer.tokenization_reformer import ReformerTokenizer
    from ..t5.tokenization_t5 import T5Tokenizer
    from ..xlm_prophetnet.tokenization_xlm_prophetnet import XLMProphetNetTokenizer
    from ..xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
    from ..xlnet.tokenization_xlnet import XLNetTokenizer
else:
    AlbertTokenizer = None
    BarthezTokenizer = None
    BertGenerationTokenizer = None
    CamembertTokenizer = None
    MarianTokenizer = None
    MBartTokenizer = None
    MT5Tokenizer = None
    PegasusTokenizer = None
    ReformerTokenizer = None
    T5Tokenizer = None
    XLMRobertaTokenizer = None
    XLNetTokenizer = None
    XLMProphetNetTokenizer = None

if is_tokenizers_available():
    from ..albert.tokenization_albert_fast import AlbertTokenizerFast
    from ..bart.tokenization_bart_fast import BartTokenizerFast
    from ..barthez.tokenization_barthez_fast import BarthezTokenizerFast
    from ..bert.tokenization_bert_fast import BertTokenizerFast
    from ..camembert.tokenization_camembert_fast import CamembertTokenizerFast
    from ..distilbert.tokenization_distilbert_fast import DistilBertTokenizerFast
    from ..dpr.tokenization_dpr_fast import DPRQuestionEncoderTokenizerFast
    from ..electra.tokenization_electra_fast import ElectraTokenizerFast
    from ..funnel.tokenization_funnel_fast import FunnelTokenizerFast
    from ..gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
    from ..herbert.tokenization_herbert_fast import HerbertTokenizerFast
    from ..layoutlm.tokenization_layoutlm_fast import LayoutLMTokenizerFast
    from ..longformer.tokenization_longformer_fast import LongformerTokenizerFast
    from ..lxmert.tokenization_lxmert_fast import LxmertTokenizerFast
    from ..mbart.tokenization_mbart_fast import MBartTokenizerFast
    from ..mobilebert.tokenization_mobilebert_fast import MobileBertTokenizerFast
    from ..mt5 import MT5TokenizerFast
    from ..openai.tokenization_openai_fast import OpenAIGPTTokenizerFast
    from ..pegasus.tokenization_pegasus_fast import PegasusTokenizerFast
    from ..reformer.tokenization_reformer_fast import ReformerTokenizerFast
    from ..retribert.tokenization_retribert_fast import RetriBertTokenizerFast
    from ..roberta.tokenization_roberta_fast import RobertaTokenizerFast
    from ..squeezebert.tokenization_squeezebert_fast import SqueezeBertTokenizerFast
    from ..t5.tokenization_t5_fast import T5TokenizerFast
    from ..xlm_roberta.tokenization_xlm_roberta_fast import XLMRobertaTokenizerFast
    from ..xlnet.tokenization_xlnet_fast import XLNetTokenizerFast
else:
    AlbertTokenizerFast = None
    BartTokenizerFast = None
    BarthezTokenizerFast = None
    BertTokenizerFast = None
    CamembertTokenizerFast = None
    DistilBertTokenizerFast = None
    DPRQuestionEncoderTokenizerFast = None
    ElectraTokenizerFast = None
    FunnelTokenizerFast = None
    GPT2TokenizerFast = None
    HerbertTokenizerFast = None
    LayoutLMTokenizerFast = None
    LongformerTokenizerFast = None
    LxmertTokenizerFast = None
    MBartTokenizerFast = None
    MobileBertTokenizerFast = None
    MT5TokenizerFast = None
    OpenAIGPTTokenizerFast = None
    PegasusTokenizerFast = None
    ReformerTokenizerFast = None
    RetriBertTokenizerFast = None
    RobertaTokenizerFast = None
    SqueezeBertTokenizerFast = None
    T5TokenizerFast = None
    XLMRobertaTokenizerFast = None
    XLNetTokenizerFast = None

logger = logging.get_logger(__name__)


TOKENIZER_MAPPING = OrderedDict(
    [
        (RetriBertConfig, (RetriBertTokenizer, RetriBertTokenizerFast)),
        (T5Config, (T5Tokenizer, T5TokenizerFast)),
        (MT5Config, (MT5Tokenizer, MT5TokenizerFast)),
        (MobileBertConfig, (MobileBertTokenizer, MobileBertTokenizerFast)),
        (DistilBertConfig, (DistilBertTokenizer, DistilBertTokenizerFast)),
        (AlbertConfig, (AlbertTokenizer, AlbertTokenizerFast)),
        (CamembertConfig, (CamembertTokenizer, CamembertTokenizerFast)),
        (PegasusConfig, (PegasusTokenizer, PegasusTokenizerFast)),
        (MBartConfig, (MBartTokenizer, MBartTokenizerFast)),
        (XLMRobertaConfig, (XLMRobertaTokenizer, XLMRobertaTokenizerFast)),
        (MarianConfig, (MarianTokenizer, None)),
        (BlenderbotConfig, (BlenderbotSmallTokenizer, None)),
        (LongformerConfig, (LongformerTokenizer, LongformerTokenizerFast)),
        (BartConfig, (BarthezTokenizer, BarthezTokenizerFast)),
        (BartConfig, (BartTokenizer, BartTokenizerFast)),
        (LongformerConfig, (LongformerTokenizer, LongformerTokenizerFast)),
        (RobertaConfig, (RobertaTokenizer, RobertaTokenizerFast)),
        (ReformerConfig, (ReformerTokenizer, ReformerTokenizerFast)),
        (ElectraConfig, (ElectraTokenizer, ElectraTokenizerFast)),
        (FunnelConfig, (FunnelTokenizer, FunnelTokenizerFast)),
        (LxmertConfig, (LxmertTokenizer, LxmertTokenizerFast)),
        (LayoutLMConfig, (LayoutLMTokenizer, LayoutLMTokenizerFast)),
        (DPRConfig, (DPRQuestionEncoderTokenizer, DPRQuestionEncoderTokenizerFast)),
        (SqueezeBertConfig, (SqueezeBertTokenizer, SqueezeBertTokenizerFast)),
        (BertConfig, (BertTokenizer, BertTokenizerFast)),
        (OpenAIGPTConfig, (OpenAIGPTTokenizer, OpenAIGPTTokenizerFast)),
        (GPT2Config, (GPT2Tokenizer, GPT2TokenizerFast)),
        (TransfoXLConfig, (TransfoXLTokenizer, None)),
        (XLNetConfig, (XLNetTokenizer, XLNetTokenizerFast)),
        (FlaubertConfig, (FlaubertTokenizer, None)),
        (XLMConfig, (XLMTokenizer, None)),
        (CTRLConfig, (CTRLTokenizer, None)),
        (FSMTConfig, (FSMTTokenizer, None)),
        (BertGenerationConfig, (BertGenerationTokenizer, None)),
        (DebertaConfig, (DebertaTokenizer, None)),
        (RagConfig, (RagTokenizer, None)),
        (XLMProphetNetConfig, (XLMProphetNetTokenizer, None)),
        (ProphetNetConfig, (ProphetNetTokenizer, None)),
    ]
)

# For tokenizers which are not directly mapped from a config
NO_CONFIG_TOKENIZER = [
    BertJapaneseTokenizer,
    BertweetTokenizer,
    HerbertTokenizer,
    HerbertTokenizerFast,
    PhobertTokenizer,
]


SLOW_TOKENIZER_MAPPING = {
    k: (v[0] if v[0] is not None else v[1])
    for k, v in TOKENIZER_MAPPING.items()
    if (v[0] is not None or v[1] is not None)
}




class AutoTokenizer:
    r"""
    This is a generic tokenizer class that will be instantiated as one of the tokenizer classes of the library when
    created with the :meth:`AutoTokenizer.from_pretrained` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """


    @classmethod
    @replace_list_option_in_docstrings(SLOW_TOKENIZER_MAPPING)