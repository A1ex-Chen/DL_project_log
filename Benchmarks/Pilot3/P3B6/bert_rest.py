import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification


class HiBERT(nn.Module):
