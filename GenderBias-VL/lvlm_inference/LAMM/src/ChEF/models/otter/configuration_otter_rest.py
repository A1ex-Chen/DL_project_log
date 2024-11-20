import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.clip import CLIPVisionConfig

logger = logging.get_logger(__name__)


class OtterConfig(PretrainedConfig):
    r"""
    [`OtterConfig`] is the configuration class to store the configuration of a [`OtterForConditionalGeneration`]. It is
    used to instantiate a Otter model according to the specified arguments, defining the vision model and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the Otter architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`PretrainedConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        cross_attn_every_n_layers (`int`, *optional*, defaults to 4):
            The number of cross-attention layers adding after each transformer layer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     PretrainedConfig,
    ...     OPTConfig,
    ...     OtterConfig,
    ...     OtterForConditionalGeneration,
    ... )

    >>> # Initializing a OtterConfig with luodian/otter-9b-hf style configuration
    >>> configuration = OtterConfig()

    >>> # Initializing a OtterForConditionalGeneration (with random weights) from the Salesforce/Otter-opt-2.7b style configuration
    >>> model = OtterForConditionalGeneration(configuration)
    ```"""
    model_type = "otter"
    is_composition = True

