# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for BLIP."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import convert_to_rgb, resize, to_channel_dimension_format
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_list_of_images,
    to_numpy_array,
    valid_images,
)
from transformers.utils import TensorType, is_vision_available, logging

from diffusers.utils import numpy_to_pil


if is_vision_available():
    import PIL.Image


logger = logging.get_logger(__name__)


# We needed some extra functions on top of the ones in transformers.image_processing_utils.BaseImageProcessor, namely center crop
# Copy-pasted from transformers.models.blip.image_processing_blip.BlipImageProcessor
class BlipImageProcessor(BaseImageProcessor):
    r"""
    Constructs a BLIP image processor.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*, defaults to `{"height": 384, "width": 384}`):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Wwhether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_MEAN`):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*, defaults to `IMAGENET_STANDARD_STD`):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
    """

    model_input_names = ["pixel_values"]


    # Copy-pasted from transformers.models.vit.image_processing_vit.ViTImageProcessor.resize with PILImageResampling.BILINEAR->PILImageResampling.BICUBIC


    # Follows diffusers.VaeImageProcessor.postprocess