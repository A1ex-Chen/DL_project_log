import logging
import os

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairseq import utils
from fairseq.data import (
    FairseqDataset,
    AppendTokenDataset,
    Dictionary,
    IdDataset,
    LMContextWindowDataset,
    MonolingualDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
    RawLabelDataset,
    TruncatedDictionary,
    data_utils,
)
from fairseq import utils
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask
from fairseq.data import Dictionary, data_utils
from omegaconf import II
from fairseq import metrics, search, tokenizer, utils
from ..data.utils import SPECIAL_SYMBOLS, add_location_symbols

import pdb

logger = logging.getLogger(__name__)

@dataclass
class GenerationObjConfig(LanguageModelingConfig):
    required_batch_size_multiple: int = II("dataset.required_batch_size_multiple")
    dict_path: str = field(
        default="",
        metadata={
            "help": "dictionary path"
        },
    )
    image_feature_length: int = field(
        default=0,
        metadata={
            "help": "image feature length"
        },
    ) 
    input_resolution: int = field(default=224, metadata={"help": ""})
    # newsetting
    location_bin_size: int = field(
        default=16, 
        metadata={
            "help": "used to discrete the continuous coordinates"
        },
    )  
    locate_special_token: int = field(
        default=0, 
        metadata={"help": "used to discrete the continuous coordinates"}
    )


class RawImageDataset(FairseqDataset):





@register_task("generation_obj", dataclass=GenerationObjConfig)
class GenerationObjTask(LanguageModelingTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @classmethod






        