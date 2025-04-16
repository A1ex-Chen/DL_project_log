from .base import BaseModel
from .bert_modules.bert import BERT

import torch.nn as nn


class BERTModel(BaseModel):

    @classmethod
