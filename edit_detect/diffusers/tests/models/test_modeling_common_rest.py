# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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

import inspect
import tempfile
import traceback
import unittest
import unittest.mock as mock
import uuid
from typing import Dict, List, Tuple

import numpy as np
import requests_mock
import torch
from accelerate.utils import compute_module_sizes
from huggingface_hub import ModelCard, delete_repo
from huggingface_hub.utils import is_jinja_available
from requests.exceptions import HTTPError

from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    AttnProcessorNPU,
    XFormersAttnProcessor,
)
from diffusers.training_utils import EMAModel
from diffusers.utils import is_torch_npu_available, is_xformers_available, logging
from diffusers.utils.testing_utils import (
    CaptureLogger,
    get_python_version,
    require_python39_or_higher,
    require_torch_2,
    require_torch_accelerator_with_training,
    require_torch_gpu,
    require_torch_multi_gpu,
    run_test_in_subprocess,
    torch_device,
)

from ..others.test_utils import TOKEN, USER, is_staging_test


# Will be run via run_test_in_subprocess


class ModelUtilsTest(unittest.TestCase):






class UNetTesterMixin:



class ModelTesterMixin:
    main_input_name = None  # overwrite in model specific tester class
    base_precision = 1e-3
    forward_requires_fresh_args = False
    model_split_percents = [0.5, 0.7, 0.9]




    @unittest.skipIf(
        torch_device != "npu" or not is_torch_npu_available(),
        reason="torch npu flash attention is only available with NPU and `torch_npu` installed",
    )

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )

    @require_torch_gpu


    @require_python39_or_higher
    @require_torch_2
    @unittest.skipIf(
        get_python_version == (3, 12),
        reason="Torch Dynamo isn't yet supported for Python 3.12.",
    )





    @require_torch_accelerator_with_training

    @require_torch_accelerator_with_training


    @require_torch_accelerator_with_training


    @require_torch_gpu

    @require_torch_gpu

    @require_torch_gpu

    @require_torch_multi_gpu


@is_staging_test
class ModelPushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-model-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"



    @unittest.skipIf(
        not is_jinja_available(),
        reason="Model card tests cannot be performed without Jinja installed.",
    )


        if self.forward_requires_fresh_args:
            model = self.model_class(**self.init_dict)
        else:
            init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
            model = self.model_class(**init_dict)

        model.to(torch_device)
        model.eval()

        with torch.no_grad():
            if self.forward_requires_fresh_args:
                outputs_dict = model(**self.inputs_dict(0))
                outputs_tuple = model(**self.inputs_dict(0), return_dict=False)
            else:
                outputs_dict = model(**inputs_dict)
                outputs_tuple = model(**inputs_dict, return_dict=False)

        recursive_check(outputs_tuple, outputs_dict)

    @require_torch_accelerator_with_training
    def test_enable_disable_gradient_checkpointing(self):
        if not self.model_class._supports_gradient_checkpointing:
            return  # Skip test if model does not support gradient checkpointing

        init_dict, _ = self.prepare_init_args_and_inputs_for_common()

        # at init model should have gradient checkpointing disabled
        model = self.model_class(**init_dict)
        self.assertFalse(model.is_gradient_checkpointing)

        # check enable works
        model.enable_gradient_checkpointing()
        self.assertTrue(model.is_gradient_checkpointing)

        # check disable works
        model.disable_gradient_checkpointing()
        self.assertFalse(model.is_gradient_checkpointing)

    def test_deprecated_kwargs(self):
        has_kwarg_in_model_class = "kwargs" in inspect.signature(self.model_class.__init__).parameters
        has_deprecated_kwarg = len(self.model_class._deprecated_kwargs) > 0

        if has_kwarg_in_model_class and not has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} has `**kwargs` in its __init__ method but has not defined any deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if there are"
                " no deprecated arguments or add the deprecated argument with `_deprecated_kwargs ="
                " [<deprecated_argument>]`"
            )

        if not has_kwarg_in_model_class and has_deprecated_kwarg:
            raise ValueError(
                f"{self.model_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated kwargs"
                " under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs` argument to"
                f" {self.model_class}.__init__ if there are deprecated arguments or remove the deprecated argument"
                " from `_deprecated_kwargs = [<deprecated_argument>]`"
            )

    @require_torch_gpu
    def test_cpu_offload(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        # We test several splits of sizes to make sure it works.
        max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir)

            for max_size in max_gpu_sizes:
                max_memory = {0: max_size, "cpu": model_size * 2}
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                # Making sure part of the model will actually end up offloaded
                self.assertSetEqual(set(new_model.hf_device_map.values()), {0, "cpu"})

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)
                torch.manual_seed(0)
                new_output = new_model(**inputs_dict)

                self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_gpu
    def test_disk_offload_without_safetensors(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir, safe_serialization=False)

            with self.assertRaises(ValueError):
                max_size = int(self.model_split_percents[0] * model_size)
                max_memory = {0: max_size, "cpu": max_size}
                # This errors out because it's missing an offload folder
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)

            max_size = int(self.model_split_percents[0] * model_size)
            max_memory = {0: max_size, "cpu": max_size}
            new_model = self.model_class.from_pretrained(
                tmp_dir, device_map="auto", max_memory=max_memory, offload_folder=tmp_dir
            )

            self.check_device_map_is_respected(new_model, new_model.hf_device_map)
            torch.manual_seed(0)
            new_output = new_model(**inputs_dict)

            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_gpu
    def test_disk_offload_with_safetensors(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir)

            max_size = int(self.model_split_percents[0] * model_size)
            max_memory = {0: max_size, "cpu": max_size}
            new_model = self.model_class.from_pretrained(
                tmp_dir, device_map="auto", offload_folder=tmp_dir, max_memory=max_memory
            )

            self.check_device_map_is_respected(new_model, new_model.hf_device_map)
            torch.manual_seed(0)
            new_output = new_model(**inputs_dict)

            self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))

    @require_torch_multi_gpu
    def test_model_parallelism(self):
        config, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        model = self.model_class(**config).eval()
        if model._no_split_modules is None:
            return

        model = model.to(torch_device)

        torch.manual_seed(0)
        base_output = model(**inputs_dict)

        model_size = compute_module_sizes(model)[""]
        # We test several splits of sizes to make sure it works.
        max_gpu_sizes = [int(p * model_size) for p in self.model_split_percents[1:]]
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.cpu().save_pretrained(tmp_dir)

            for max_size in max_gpu_sizes:
                max_memory = {0: max_size, 1: model_size * 2, "cpu": model_size * 2}
                new_model = self.model_class.from_pretrained(tmp_dir, device_map="auto", max_memory=max_memory)
                # Making sure part of the model will actually end up offloaded
                self.assertSetEqual(set(new_model.hf_device_map.values()), {0, 1})

                self.check_device_map_is_respected(new_model, new_model.hf_device_map)

                torch.manual_seed(0)
                new_output = new_model(**inputs_dict)

                self.assertTrue(torch.allclose(base_output[0], new_output[0], atol=1e-5))


@is_staging_test
class ModelPushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-model-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def test_push_to_hub(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id=self.repo_id, push_to_hub=True, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(f"{USER}/{self.repo_id}")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)

    def test_push_to_hub_in_organization(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.org_repo_id, token=TOKEN)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, push_to_hub=True, token=TOKEN, repo_id=self.org_repo_id)

        new_model = UNet2DConditionModel.from_pretrained(self.org_repo_id)
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(self.org_repo_id, token=TOKEN)

    @unittest.skipIf(
        not is_jinja_available(),
        reason="Model card tests cannot be performed without Jinja installed.",
    )
    def test_push_to_hub_library_name(self):
        model = UNet2DConditionModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            cross_attention_dim=32,
        )
        model.push_to_hub(self.repo_id, token=TOKEN)

        model_card = ModelCard.load(f"{USER}/{self.repo_id}", token=TOKEN).data
        assert model_card.library_name == "diffusers"

        # Reset repo
        delete_repo(self.repo_id, token=TOKEN)