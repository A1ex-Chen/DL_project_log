import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from ...models.controlnet import ControlNetModel, ControlNetOutput
from ...models.modeling_utils import ModelMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class MultiControlNetModel(ModelMixin):
    r"""
    Multiple `ControlNetModel` wrapper class for Multi-ControlNet

    This module is a wrapper for multiple instances of the `ControlNetModel`. The `forward()` API is designed to be
    compatible with `ControlNetModel`.

    Args:
        controlnets (`List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `ControlNetModel` as a list.
    """




    @classmethod