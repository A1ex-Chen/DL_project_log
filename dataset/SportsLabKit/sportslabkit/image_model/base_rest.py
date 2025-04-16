from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from sportslabkit.types.detection import Detection
from sportslabkit.types.detections import Detections
from sportslabkit.utils import read_image


class BaseImageModel(ABC):
    """
    Base class for image embedding models. This class implements basic functionality for handling input and output data, and requires subclasses to implement model loading and forward pass functionality.

    Subclasses should override the 'load' and 'forward' methods. The 'load' method should handle loading the model from the specified repository and checkpoint, and 'forward' should define the forward pass of the model. Then add `ConfigTemplates` for your model to define the available configuration options.

    The input to the model should be flexible. It accepts numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays representing the images.

    The output of the model is expected to be a list of embeddings, one for each image. If the model's output does not meet this expectation, `_check_and_fix_outputs` method should convert the output into a compatible format.

    Example:
        class CustomImageModel(BaseImageModel):
            def load(self):
                # Load your model here
                pass

            def forward(self, x):
                # Define the forward pass here
                pass

    Attributes:
        model_config (Optional[dict]): The configuration for the model.
        inference_config (Optional[dict]): The configuration for the inference.
        input_is_batched (bool): Whether the input is batched or not. This is set by the `_check_and_fix_inputs` method.
    """






    @abstractmethod
