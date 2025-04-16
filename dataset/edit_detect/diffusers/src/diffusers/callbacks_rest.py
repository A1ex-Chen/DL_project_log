from typing import Any, Dict, List

from .configuration_utils import ConfigMixin, register_to_config
from .utils import CONFIG_NAME


class PipelineCallback(ConfigMixin):
    """
    Base class for all the official callbacks used in a pipeline. This class provides a structure for implementing
    custom callbacks and ensures that all callbacks have a consistent interface.

    Please implement the following:
        `tensor_inputs`: This should return a list of tensor inputs specific to your callback. You will only be able to
        include
            variables listed in the `._callback_tensor_inputs` attribute of your pipeline class.
        `callback_fn`: This method defines the core functionality of your callback.
    """

    config_name = CONFIG_NAME

    @register_to_config

    @property




class MultiPipelineCallbacks:
    """
    This class is designed to handle multiple pipeline callbacks. It accepts a list of PipelineCallback objects and
    provides a unified interface for calling all of them.
    """


    @property



class SDCFGCutoffCallback(PipelineCallback):
    """
    Callback function for Stable Diffusion Pipelines. After certain number of steps (set by `cutoff_step_ratio` or
    `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """

    tensor_inputs = ["prompt_embeds"]



class SDXLCFGCutoffCallback(PipelineCallback):
    """
    Callback function for Stable Diffusion XL Pipelines. After certain number of steps (set by `cutoff_step_ratio` or
    `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """

    tensor_inputs = ["prompt_embeds", "add_text_embeds", "add_time_ids"]



class IPAdapterScaleCutoffCallback(PipelineCallback):
    """
    Callback function for any pipeline that inherits `IPAdapterMixin`. After certain number of steps (set by
    `cutoff_step_ratio` or `cutoff_step_index`), this callback will set the IP Adapter scale to `0.0`.

    Note: This callback mutates the IP Adapter attention processors by setting the scale to 0.0 after the cutoff step.
    """

    tensor_inputs = []
