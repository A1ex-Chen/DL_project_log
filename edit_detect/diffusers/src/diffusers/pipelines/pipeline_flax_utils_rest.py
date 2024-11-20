# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import inspect
import os
from typing import Any, Dict, List, Optional, Union

import flax
import numpy as np
import PIL.Image
from flax.core.frozen_dict import FrozenDict
from huggingface_hub import create_repo, snapshot_download
from huggingface_hub.utils import validate_hf_hub_args
from PIL import Image
from tqdm.auto import tqdm

from ..configuration_utils import ConfigMixin
from ..models.modeling_flax_utils import FLAX_WEIGHTS_NAME, FlaxModelMixin
from ..schedulers.scheduling_utils_flax import SCHEDULER_CONFIG_NAME, FlaxSchedulerMixin
from ..utils import (
    CONFIG_NAME,
    BaseOutput,
    PushToHubMixin,
    http_user_agent,
    is_transformers_available,
    logging,
)


if is_transformers_available():
    from transformers import FlaxPreTrainedModel

INDEX_FILE = "diffusion_flax_model.bin"


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "FlaxModelMixin": ["save_pretrained", "from_pretrained"],
        "FlaxSchedulerMixin": ["save_pretrained", "from_pretrained"],
        "FlaxDiffusionPipeline": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "FlaxPreTrainedModel": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])




@flax.struct.dataclass
class FlaxImagePipelineOutput(BaseOutput):
    """
    Output class for image pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class FlaxDiffusionPipeline(ConfigMixin, PushToHubMixin):
    r"""
    Base class for Flax-based pipelines.

    [`FlaxDiffusionPipeline`] stores all components (models, schedulers, and processors) for diffusion pipelines and
    provides methods for loading, downloading and saving models. It also includes methods to:

        - enable/disable the progress bar for the denoising iteration

    Class attributes:

        - **config_name** ([`str`]) -- The configuration filename that stores the class and module names of all the
          diffusion pipeline's components.
    """

    config_name = "model_index.json"



    @classmethod
    @validate_hf_hub_args

    @classmethod

    @property

    @staticmethod

    # TODO: make it compatible with jax.lax


        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        # Throw nice warnings / errors for fast accelerate loading
        if len(unused_kwargs) > 0:
            logger.warning(
                f"Keyword arguments {unused_kwargs} are not expected by {pipeline_class.__name__} and will be ignored."
            )

        # inference_params
        params = {}

        # import it here to avoid circular import
        from diffusers import pipelines

        # 3. Load each module in the pipeline
        for name, (library_name, class_name) in init_dict.items():
            if class_name is None:
                # edge case for when the pipeline was saved with safety_checker=None
                init_kwargs[name] = None
                continue

            is_pipeline_module = hasattr(pipelines, library_name)
            loaded_sub_model = None
            sub_model_should_be_defined = True

            # if the model is in a pipeline module, then we load it from the pipeline
            if name in passed_class_obj:
                # 1. check that passed_class_obj has correct parent class
                if not is_pipeline_module:
                    library = importlib.import_module(library_name)
                    class_obj = getattr(library, class_name)
                    importable_classes = LOADABLE_CLASSES[library_name]
                    class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

                    expected_class_obj = None
                    for class_name, class_candidate in class_candidates.items():
                        if class_candidate is not None and issubclass(class_obj, class_candidate):
                            expected_class_obj = class_candidate

                    if not issubclass(passed_class_obj[name].__class__, expected_class_obj):
                        raise ValueError(
                            f"{passed_class_obj[name]} is of type: {type(passed_class_obj[name])}, but should be"
                            f" {expected_class_obj}"
                        )
                elif passed_class_obj[name] is None:
                    logger.warning(
                        f"You have passed `None` for {name} to disable its functionality in {pipeline_class}. Note"
                        f" that this might lead to problems when using {pipeline_class} and is not recommended."
                    )
                    sub_model_should_be_defined = False
                else:
                    logger.warning(
                        f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
                        " has the correct type"
                    )

                # set passed class object
                loaded_sub_model = passed_class_obj[name]
            elif is_pipeline_module:
                pipeline_module = getattr(pipelines, library_name)
                class_obj = import_flax_or_no_model(pipeline_module, class_name)

                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {c: class_obj for c in importable_classes.keys()}
            else:
                # else we just import it from the library.
                library = importlib.import_module(library_name)
                class_obj = import_flax_or_no_model(library, class_name)

                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

            if loaded_sub_model is None and sub_model_should_be_defined:
                load_method_name = None
                for class_name, class_candidate in class_candidates.items():
                    if class_candidate is not None and issubclass(class_obj, class_candidate):
                        load_method_name = importable_classes[class_name][1]

                load_method = getattr(class_obj, load_method_name)

                # check if the module is in a subdirectory
                if os.path.isdir(os.path.join(cached_folder, name)):
                    loadable_folder = os.path.join(cached_folder, name)
                else:
                    loaded_sub_model = cached_folder

                if issubclass(class_obj, FlaxModelMixin):
                    loaded_sub_model, loaded_params = load_method(
                        loadable_folder,
                        from_pt=from_pt,
                        use_memory_efficient_attention=use_memory_efficient_attention,
                        split_head_dim=split_head_dim,
                        dtype=dtype,
                    )
                    params[name] = loaded_params
                elif is_transformers_available() and issubclass(class_obj, FlaxPreTrainedModel):
                    if from_pt:
                        # TODO(Suraj): Fix this in Transformers. We should be able to use `_do_init=False` here
                        loaded_sub_model = load_method(loadable_folder, from_pt=from_pt)
                        loaded_params = loaded_sub_model.params
                        del loaded_sub_model._params
                    else:
                        loaded_sub_model, loaded_params = load_method(loadable_folder, _do_init=False)
                    params[name] = loaded_params
                elif issubclass(class_obj, FlaxSchedulerMixin):
                    loaded_sub_model, scheduler_state = load_method(loadable_folder)
                    params[name] = scheduler_state
                else:
                    loaded_sub_model = load_method(loadable_folder)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        # 4. Potentially add passed objects if expected
        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())

        if len(missing_modules) > 0 and missing_modules <= set(passed_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        model = pipeline_class(**init_kwargs, dtype=dtype)
        return model, params

    @classmethod
    def _get_signature_keys(cls, obj):
        parameters = inspect.signature(obj.__init__).parameters
        required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
        optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
        expected_modules = set(required_parameters.keys()) - {"self"}

        return expected_modules, optional_parameters

    @property
    def components(self) -> Dict[str, Any]:
        r"""

        The `self.components` property can be useful to run different pipelines with the same weights and
        configurations to not have to re-allocate memory.

        Examples:

        ```py
        >>> from diffusers import (
        ...     FlaxStableDiffusionPipeline,
        ...     FlaxStableDiffusionImg2ImgPipeline,
        ... )

        >>> text2img = FlaxStableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", revision="bf16", dtype=jnp.bfloat16
        ... )
        >>> img2img = FlaxStableDiffusionImg2ImgPipeline(**text2img.components)
        ```

        Returns:
            A dictionary containing all the modules needed to initialize the pipeline.
        """
        expected_modules, optional_parameters = self._get_signature_keys(self)
        components = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }

        if set(components.keys()) != expected_modules:
            raise ValueError(
                f"{self} has been incorrectly initialized or {self.__class__} is incorrectly implemented. Expected"
                f" {expected_modules} to be defined, but {components} are defined."
            )

        return components

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a NumPy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    # TODO: make it compatible with jax.lax
    def progress_bar(self, iterable):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        return tqdm(iterable, **self._progress_bar_config)

    def set_progress_bar_config(self, **kwargs):
        self._progress_bar_config = kwargs