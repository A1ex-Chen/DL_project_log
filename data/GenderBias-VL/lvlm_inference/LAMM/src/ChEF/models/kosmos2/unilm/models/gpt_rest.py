from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from torch import Tensor

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import distributed_utils, utils
from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.utils import safe_getattr, safe_hasattr

from fairseq.models import (
  BaseFairseqModel,
  register_model,
  register_model_architecture,
)
from fairseq.models.transformer_lm import (
  TransformerLanguageModelConfig,
  TransformerLanguageModel,
  base_gpt3_architecture,
)
from fairseq.models.transformer.transformer_decoder import TransformerDecoder
from fairseq.models import (
    FairseqIncrementalDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
from fairseq.modules import PositionalEmbedding
from omegaconf import II

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder

from torchscale.architecture.config import DecoderConfig
from torchscale.architecture.decoder import Decoder
from torchscale.component.embedding import TextEmbedding

logger = logging.getLogger(__name__)
DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class GPTModelConfig(TransformerLanguageModelConfig):
    scale_final_logits: bool = field(
        default=False,
        metadata={
            "help": "scale final logits by sqrt(d)"
        },
    )

    gpt_model_path: str = field(
        default="",
        metadata={"help": "gpt checkpoint path"},
    )
    rescale_init: bool = field(
        default=False,
        metadata={
            "help": "whether to use rescale initialization"
        },
    )
    deepnet: bool = field(
        default=False,
        metadata={
            "help": "enable deepnet in decoder"
        },
    )
    last_ln_scale: bool = field(
        default=False,
        metadata={
            "help": "enable last_ln_scale in decoder"
        },
    )

    # options from other parts of the config
    add_bos_token: bool = II("task.add_bos_token")
    tokens_per_sample: int = II("task.tokens_per_sample")
    max_target_positions: Optional[int] = II("task.max_target_positions")
    tpu: bool = II("common.tpu")
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    fp16: bool = II("common.fp16")
    fp16_no_flatten_grads: bool = II("common.fp16_no_flatten_grads")
    ddp_backend: str = II("distributed_training.ddp_backend")
    world_size: int = II("distributed_training.distributed_world_size")
    distributed_rank: int = II("distributed_training.distributed_rank")
    ddp_rank: int = II("distributed_training.distributed_rank")
    
    deepnorm: Optional[bool] = field(
        default=False,
    )
    subln: Optional[bool] = field(
        default=False,
    )
    rel_pos_buckets: Optional[int] = field(
        default=0,
    )
    max_rel_pos: Optional[int] = field(
        default=0,
    )
    flash_attention: bool = field(
        default=False,
    )
    sope_rel_pos: Optional[bool] = field(
        default=False,
        metadata={"help": "use SoPE as the relative position embhedding"},
    )
    scale_length: Optional[int] = field(
        default=2048,
    )
    max_chunk_emb: Optional[int] = field(
        default=0,
        metadata={"help": "chunk embedding, text image text image text: 0, 1, 1, 2, 2"},
    )
    segment_emb: Optional[bool] = field(
        default=False,
    )

@register_model("gptmodel", dataclass=GPTModelConfig)
class GPTmodel(TransformerLanguageModel):

    @classmethod

    @classmethod


class LMDecoder(Decoder, FairseqIncrementalDecoder):



