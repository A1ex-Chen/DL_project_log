# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError, validate_hf_hub_args
from packaging import version

from ..utils import deprecate, is_transformers_available, logging
from .single_file_utils import (
    SingleFileComponentError,
    _is_model_weights_in_cached_folder,
    _legacy_load_clip_tokenizer,
    _legacy_load_safety_checker,
    _legacy_load_scheduler,
    create_diffusers_clip_model_from_ldm,
    fetch_diffusers_config,
    fetch_original_config,
    is_clip_model_in_single_file,
    load_single_file_checkpoint,
)


logger = logging.get_logger(__name__)

# Legacy behaviour. `from_single_file` does not load the safety checker unless explicitly provided
SINGLE_FILE_OPTIONAL_COMPONENTS = ["safety_checker"]


if is_transformers_available():
    import transformers
    from transformers import PreTrainedModel, PreTrainedTokenizer










class FromSingleFileMixin:
    """
    Load model weights saved in the `.ckpt` format into a [`DiffusionPipeline`].
    """

    @classmethod
    @validate_hf_hub_args

        init_dict = {k: v for k, v in init_dict.items() if load_module(k, v)}

        for name, (library_name, class_name) in logging.tqdm(
            sorted(init_dict.items()), desc="Loading pipeline components..."
        ):
            loaded_sub_model = None
            is_pipeline_module = hasattr(pipelines, library_name)

            if name in passed_class_obj:
                loaded_sub_model = passed_class_obj[name]

            else:
                try:
                    loaded_sub_model = load_single_file_sub_model(
                        library_name=library_name,
                        class_name=class_name,
                        name=name,
                        checkpoint=checkpoint,
                        is_pipeline_module=is_pipeline_module,
                        cached_model_config_path=cached_model_config_path,
                        pipelines=pipelines,
                        torch_dtype=torch_dtype,
                        original_config=original_config,
                        local_files_only=local_files_only,
                        is_legacy_loading=is_legacy_loading,
                        **kwargs,
                    )
                except SingleFileComponentError as e:
                    raise SingleFileComponentError(
                        (
                            f"{e.message}\n"
                            f"Please load the component before passing it in as an argument to `from_single_file`.\n"
                            f"\n"
                            f"{name} = {class_name}.from_pretrained('...')\n"
                            f"pipe = {pipeline_class.__name__}.from_single_file(<checkpoint path>, {name}={name})\n"
                            f"\n"
                        )
                    )

            init_kwargs[name] = loaded_sub_model

        missing_modules = set(expected_modules) - set(init_kwargs.keys())
        passed_modules = list(passed_class_obj.keys())
        optional_modules = pipeline_class._optional_components

        if len(missing_modules) > 0 and missing_modules <= set(passed_modules + optional_modules):
            for module in missing_modules:
                init_kwargs[module] = passed_class_obj.get(module, None)
        elif len(missing_modules) > 0:
            passed_modules = set(list(init_kwargs.keys()) + list(passed_class_obj.keys())) - optional_kwargs
            raise ValueError(
                f"Pipeline {pipeline_class} expected {expected_modules}, but only {passed_modules} were passed."
            )

        # deprecated kwargs
        load_safety_checker = kwargs.pop("load_safety_checker", None)
        if load_safety_checker is not None:
            deprecation_message = (
                "Please pass instances of `StableDiffusionSafetyChecker` and `AutoImageProcessor`"
                "using the `safety_checker` and `feature_extractor` arguments in `from_single_file`"
            )
            deprecate("load_safety_checker", "1.0.0", deprecation_message)

            safety_checker_components = _legacy_load_safety_checker(local_files_only, torch_dtype)
            init_kwargs.update(safety_checker_components)

        pipe = pipeline_class(**init_kwargs)

        if torch_dtype is not None:
            pipe.to(dtype=torch_dtype)

        return pipe