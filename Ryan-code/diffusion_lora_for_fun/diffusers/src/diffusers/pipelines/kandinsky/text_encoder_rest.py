import torch
from transformers import PreTrainedModel, XLMRobertaConfig, XLMRobertaModel


class MCLIPConfig(XLMRobertaConfig):
    model_type = "M-CLIP"



class MultilingualCLIP(PreTrainedModel):
    config_class = MCLIPConfig

