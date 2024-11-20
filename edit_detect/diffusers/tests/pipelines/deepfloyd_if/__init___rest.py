import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer, T5EncoderModel

from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import AttnAddedKVProcessor
from diffusers.pipelines.deepfloyd_if import IFWatermarker
from diffusers.utils.testing_utils import torch_device

from ..test_pipelines_common import to_np


# WARN: the hf-internal-testing/tiny-random-t5 text encoder has some non-determinism in the `save_load` tests.


class IFPipelineTesterMixin:


    # this test is modified from the base class because if pipelines set the text encoder
    # as optional with the intention that the user is allowed to encode the prompt once
    # and then pass the embeddings directly to the pipeline. The base class test uses
    # the unmodified arguments from `self.get_dummy_inputs` which will pass the unencoded
    # prompt to the pipeline when the text encoder is set to None, throwing an error.
    # So we make the test reflect the intended usage of setting the text encoder to None.

    # Modified from `PipelineTesterMixin` to set the attn processor as it's not serialized.
    # This should be handled in the base test and then this method can be removed.