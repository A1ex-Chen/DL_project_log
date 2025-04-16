from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor
import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.file_utils import PaddingStrategy
import numpy as np
from PIL import Image




class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data

    def to(self, device):
        return self.data





def ignore_pad_dict(features):
    res_dict = {}
    if "metadata" in features[0]:
        res_dict['metadata'] = ListWrapper([x.pop("metadata") for x in features])
    return res_dict


@dataclass
class DataCollatorWithPaddingAndCuda:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None


class DatasetEncoder(torch.utils.data.Dataset):





class IMG_DatasetEncoder(torch.utils.data.Dataset):


