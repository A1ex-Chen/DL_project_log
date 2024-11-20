import copy
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import load_tf_weights_in_bert
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    PreTrainedModel,
    BertEmbeddings,
    BertAttention,
    BertLayer,
    BertEncoder,
    BertOutput,
    BertPooler,
    BertLMPredictionHead,
    BertPredictionHeadTransform
)
from .configuration_coke_bert import CokeBertConfig


class CokeBertIntermediate(nn.Module):


class CokeBertOutput(nn.Module):


class DynamicKnowledgeContextEncoderLayer(nn.Module):

        
class KnowledgeFusionLayer(nn.Module):
    

class DynamicKnowledgeContextEncoder(nn.Module):


class KnowledgeFusionEncoder(nn.Module):


class CokeBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CokeBertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "coke_bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]



class CokeBertModel(CokeBertPreTrainedModel):

        # return BaseModelOutputWithPoolingAndCrossAttentions(
        #     last_hidden_state=sequence_output,
        #     pooler_output=pooled_output,
        #     past_key_values=encoder_outputs.past_key_values,
        #     hidden_states=encoder_outputs.hidden_states,
        #     attentions=encoder_outputs.attentions,
        #     cross_attentions=encoder_outputs.cross_attentions,
        # )

class WordGraphAttention(nn.Module):




class CokeBertEntPredictionHeadTransform(nn.Module):


class CokeBertEntPredictionHead(nn.Module):


class CokeBertPreTrainingHeads(nn.Module):


class CokeBertForPreTraining(CokeBertPreTrainedModel):



class CokeBertForRelationClassification(CokeBertPreTrainedModel):
