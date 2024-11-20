# Copyright (c) Alibaba.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import os
from typing import Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING
from .configuration_qwen import QWenConfig

class LlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlamaModel`]. It is used to instantiate an LLaMA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the LLaMA-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LlamaModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Llama 1 supports up to 2048 tokens,
            Llama 2 up to 4096, CodeLlama up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        attention_bias (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.


    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # Initializing a LLaMA llama-7b style configuration
    >>> configuration = LlamaConfig()

    >>> # Initializing a model from the llama-7b style configuration
    >>> model = LlamaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]



            
class MplugOwlVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MplugOwlVisionModel`]. It is used to instantiate
    a
     mPLUG-Owl vision encoder according to the specified arguments, defining the model architecture. Instantiating a
     configuration defaults will yield a similar configuration to that of the mPLUG-Owl
     [x-plug/x_plug-llama-7b](https://huggingface.co/x-plug/x_plug-llama-7b) architecture.

     Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
     documentation from [`PretrainedConfig`] for more information.

     Args:
         hidden_size (`int`, *optional*, defaults to 768):
             Dimensionality of the encoder layers and the pooler layer.
         intermediate_size (`int`, *optional*, defaults to 3072):
             Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
         num_hidden_layers (`int`, *optional*, defaults to 12):
             Number of hidden layers in the Transformer encoder.
         num_attention_heads (`int`, *optional*, defaults to 12):
             Number of attention heads for each attention layer in the Transformer encoder.
         image_size (`int`, *optional*, defaults to 224):
             The size (resolution) of each image.
         patch_size (`int`, *optional*, defaults to 32):
             The size (resolution) of each patch.
         hidden_act (`str` or `function`, *optional*, defaults to `"quick_gelu"`):
             The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
             `"relu"`, `"selu"` and `"gelu_new"` ``"quick_gelu"` are supported.
         layer_norm_eps (`float`, *optional*, defaults to 1e-5):
             The epsilon used by the layer normalization layers.
         attention_dropout (`float`, *optional*, defaults to 0.0):
             The dropout ratio for the attention probabilities.
         initializer_range (`float`, *optional*, defaults to 0.02):
             The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
         initializer_factor (`float`, *optional*, defaults to 1):
             A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
             testing).


     ```"""

    model_type = "mplug_owl_vision_model"


    @classmethod


class MplugOwlVisualAbstractorConfig(PretrainedConfig):
    model_type = "mplug_owl_visual_abstract"


    @classmethod



DEFAULT_VISUAL_CONFIG = {
    "visual_model": MplugOwlVisionConfig().to_dict(),
    "visual_abstractor": MplugOwlVisualAbstractorConfig().to_dict()
}

class MPLUGOwl2Config(LlamaConfig):
    model_type = "mplug_owl2"

class MPLUGOwl2QwenConfig(QWenConfig):
    model_type = "mplug_owl2_1"
if __name__ == "__main__":
    print(MplugOwlVisionConfig().to_dict())