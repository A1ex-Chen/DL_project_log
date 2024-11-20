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
import os
import tempfile
import unittest
from itertools import product

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils.import_utils import is_peft_available
from diffusers.utils.testing_utils import (
    floats_tensor,
    require_peft_backend,
    require_peft_version_greater,
    skip_mps,
    torch_device,
)


if is_peft_available():
    from peft import LoraConfig
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils import get_peft_model_state_dict






@require_peft_backend
class PeftLoraLoaderMixinTests:
    pipeline_class = None
    scheduler_cls = None
    scheduler_kwargs = None
    has_two_text_encoders = False
    unet_kwargs = None
    vae_kwargs = None



    # copied from: https://colab.research.google.com/gist/sayakpaul/df2ef6e1ae6d8c10a49d859883b10860/scratchpad.ipynb



















    @skip_mps



    @require_peft_version_greater(peft_version="0.6.2")

    @require_peft_version_greater(peft_version="0.9.0")

    @unittest.skip("This is failing for now - need to investigate")



        components, text_lora_config, unet_lora_config = self.get_dummy_components(self.scheduler_cls)
        pipe = self.pipeline_class(**components)
        pipe = pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)
        _, _, inputs = self.get_dummy_inputs(with_generator=False)

        pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
        pipe.unet.add_adapter(unet_lora_config, "adapter-1")

        if self.has_two_text_encoders:
            pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")

        for scale_dict in all_possible_dict_opts(pipe.unet, value=1234):
            # test if lora block scales can be set with this scale_dict
            if not self.has_two_text_encoders and "text_encoder_2" in scale_dict:
                del scale_dict["text_encoder_2"]

            pipe.set_adapters("adapter-1", scale_dict)  # test will fail if this line throws an error

    def test_simple_inference_with_text_unet_multi_adapter_delete_adapter(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set/delete them
        """
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")

            pipe.unet.add_adapter(unet_lora_config, "adapter-1")
            pipe.unet.add_adapter(unet_lora_config, "adapter-2")

            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            self.assertTrue(check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet")

            if self.has_two_text_encoders:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

            pipe.set_adapters("adapter-1")

            output_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.set_adapters("adapter-2")
            output_adapter_2 = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.set_adapters(["adapter-1", "adapter-2"])

            output_adapter_mixed = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertFalse(
                np.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
                "Adapter 1 and 2 should give different results",
            )

            self.assertFalse(
                np.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3),
                "Adapter 1 and mixed adapters should give different results",
            )

            self.assertFalse(
                np.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3),
                "Adapter 2 and mixed adapters should give different results",
            )

            pipe.delete_adapters("adapter-1")
            output_deleted_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertTrue(
                np.allclose(output_deleted_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
                "Adapter 1 and 2 should give different results",
            )

            pipe.delete_adapters("adapter-2")
            output_deleted_adapters = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertTrue(
                np.allclose(output_no_lora, output_deleted_adapters, atol=1e-3, rtol=1e-3),
                "output with no lora and output with lora disabled should give same results",
            )

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")

            pipe.unet.add_adapter(unet_lora_config, "adapter-1")
            pipe.unet.add_adapter(unet_lora_config, "adapter-2")

            pipe.set_adapters(["adapter-1", "adapter-2"])
            pipe.delete_adapters(["adapter-1", "adapter-2"])

            output_deleted_adapters = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertTrue(
                np.allclose(output_no_lora, output_deleted_adapters, atol=1e-3, rtol=1e-3),
                "output with no lora and output with lora disabled should give same results",
            )

    def test_simple_inference_with_text_unet_multi_adapter_weighted(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, attaches
        multiple adapters and set them
        """
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")

            pipe.unet.add_adapter(unet_lora_config, "adapter-1")
            pipe.unet.add_adapter(unet_lora_config, "adapter-2")

            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            self.assertTrue(check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet")

            if self.has_two_text_encoders:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

            pipe.set_adapters("adapter-1")

            output_adapter_1 = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.set_adapters("adapter-2")
            output_adapter_2 = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.set_adapters(["adapter-1", "adapter-2"])

            output_adapter_mixed = pipe(**inputs, generator=torch.manual_seed(0)).images

            # Fuse and unfuse should lead to the same results
            self.assertFalse(
                np.allclose(output_adapter_1, output_adapter_2, atol=1e-3, rtol=1e-3),
                "Adapter 1 and 2 should give different results",
            )

            self.assertFalse(
                np.allclose(output_adapter_1, output_adapter_mixed, atol=1e-3, rtol=1e-3),
                "Adapter 1 and mixed adapters should give different results",
            )

            self.assertFalse(
                np.allclose(output_adapter_2, output_adapter_mixed, atol=1e-3, rtol=1e-3),
                "Adapter 2 and mixed adapters should give different results",
            )

            pipe.set_adapters(["adapter-1", "adapter-2"], [0.5, 0.6])
            output_adapter_mixed_weighted = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertFalse(
                np.allclose(output_adapter_mixed_weighted, output_adapter_mixed, atol=1e-3, rtol=1e-3),
                "Weighted adapter and mixed adapter should give different results",
            )

            pipe.disable_lora()

            output_disabled = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertTrue(
                np.allclose(output_no_lora, output_disabled, atol=1e-3, rtol=1e-3),
                "output with no lora and output with lora disabled should give same results",
            )

    @skip_mps
    def test_lora_fuse_nan(self):
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")

            pipe.unet.add_adapter(unet_lora_config, "adapter-1")

            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            self.assertTrue(check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet")

            # corrupt one LoRA weight with `inf` values
            with torch.no_grad():
                pipe.unet.mid_block.attentions[0].transformer_blocks[0].attn1.to_q.lora_A["adapter-1"].weight += float(
                    "inf"
                )

            # with `safe_fusing=True` we should see an Error
            with self.assertRaises(ValueError):
                pipe.fuse_lora(safe_fusing=True)

            # without we should not see an error, but every image will be black
            pipe.fuse_lora(safe_fusing=False)

            out = pipe("test", num_inference_steps=2, output_type="np").images

            self.assertTrue(np.isnan(out).all())

    def test_get_adapters(self):
        """
        Tests a simple usecase where we attach multiple adapters and check if the results
        are the expected results
        """
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.unet.add_adapter(unet_lora_config, "adapter-1")

            adapter_names = pipe.get_active_adapters()
            self.assertListEqual(adapter_names, ["adapter-1"])

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            pipe.unet.add_adapter(unet_lora_config, "adapter-2")

            adapter_names = pipe.get_active_adapters()
            self.assertListEqual(adapter_names, ["adapter-2"])

            pipe.set_adapters(["adapter-1", "adapter-2"])
            self.assertListEqual(pipe.get_active_adapters(), ["adapter-1", "adapter-2"])

    def test_get_list_adapters(self):
        """
        Tests a simple usecase where we attach multiple adapters and check if the results
        are the expected results
        """
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.unet.add_adapter(unet_lora_config, "adapter-1")

            adapter_names = pipe.get_list_adapters()
            self.assertDictEqual(adapter_names, {"text_encoder": ["adapter-1"], "unet": ["adapter-1"]})

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            pipe.unet.add_adapter(unet_lora_config, "adapter-2")

            adapter_names = pipe.get_list_adapters()
            self.assertDictEqual(
                adapter_names, {"text_encoder": ["adapter-1", "adapter-2"], "unet": ["adapter-1", "adapter-2"]}
            )

            pipe.set_adapters(["adapter-1", "adapter-2"])
            self.assertDictEqual(
                pipe.get_list_adapters(),
                {"unet": ["adapter-1", "adapter-2"], "text_encoder": ["adapter-1", "adapter-2"]},
            )

            pipe.unet.add_adapter(unet_lora_config, "adapter-3")
            self.assertDictEqual(
                pipe.get_list_adapters(),
                {"unet": ["adapter-1", "adapter-2", "adapter-3"], "text_encoder": ["adapter-1", "adapter-2"]},
            )

    @require_peft_version_greater(peft_version="0.6.2")
    def test_simple_inference_with_text_lora_unet_fused_multi(self):
        """
        Tests a simple inference with lora attached into text encoder + fuses the lora weights into base model
        and makes sure it works as expected - with unet and multi-adapter case
        """
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
            self.assertTrue(output_no_lora.shape == (1, 64, 64, 3))

            pipe.text_encoder.add_adapter(text_lora_config, "adapter-1")
            pipe.unet.add_adapter(unet_lora_config, "adapter-1")

            # Attach a second adapter
            pipe.text_encoder.add_adapter(text_lora_config, "adapter-2")
            pipe.unet.add_adapter(unet_lora_config, "adapter-2")

            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            self.assertTrue(check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet")

            if self.has_two_text_encoders:
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-1")
                pipe.text_encoder_2.add_adapter(text_lora_config, "adapter-2")
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

            # set them to multi-adapter inference mode
            pipe.set_adapters(["adapter-1", "adapter-2"])
            ouputs_all_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.set_adapters(["adapter-1"])
            ouputs_lora_1 = pipe(**inputs, generator=torch.manual_seed(0)).images

            pipe.fuse_lora(adapter_names=["adapter-1"])

            # Fusing should still keep the LoRA layers so outpout should remain the same
            outputs_lora_1_fused = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertTrue(
                np.allclose(ouputs_lora_1, outputs_lora_1_fused, atol=1e-3, rtol=1e-3),
                "Fused lora should not change the output",
            )

            pipe.unfuse_lora()
            pipe.fuse_lora(adapter_names=["adapter-2", "adapter-1"])

            # Fusing should still keep the LoRA layers
            output_all_lora_fused = pipe(**inputs, generator=torch.manual_seed(0)).images
            self.assertTrue(
                np.allclose(output_all_lora_fused, ouputs_all_lora, atol=1e-3, rtol=1e-3),
                "Fused lora should not change the output",
            )

    @require_peft_version_greater(peft_version="0.9.0")
    def test_simple_inference_with_dora(self):
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls, use_dora=True)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            output_no_dora_lora = pipe(**inputs, generator=torch.manual_seed(0)).images
            self.assertTrue(output_no_dora_lora.shape == (1, 64, 64, 3))

            pipe.text_encoder.add_adapter(text_lora_config)
            pipe.unet.add_adapter(unet_lora_config)

            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            self.assertTrue(check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet")

            if self.has_two_text_encoders:
                pipe.text_encoder_2.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

            output_dora_lora = pipe(**inputs, generator=torch.manual_seed(0)).images

            self.assertFalse(
                np.allclose(output_dora_lora, output_no_dora_lora, atol=1e-3, rtol=1e-3),
                "DoRA lora should change the output",
            )

    @unittest.skip("This is failing for now - need to investigate")
    def test_simple_inference_with_text_unet_lora_unfused_torch_compile(self):
        """
        Tests a simple inference with lora attached to text encoder and unet, then unloads the lora weights
        and makes sure it works as expected
        """
        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, text_lora_config, unet_lora_config = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _, _, inputs = self.get_dummy_inputs(with_generator=False)

            pipe.text_encoder.add_adapter(text_lora_config)
            pipe.unet.add_adapter(unet_lora_config)

            self.assertTrue(check_if_lora_correctly_set(pipe.text_encoder), "Lora not correctly set in text encoder")
            self.assertTrue(check_if_lora_correctly_set(pipe.unet), "Lora not correctly set in Unet")

            if self.has_two_text_encoders:
                pipe.text_encoder_2.add_adapter(text_lora_config)
                self.assertTrue(
                    check_if_lora_correctly_set(pipe.text_encoder_2), "Lora not correctly set in text encoder 2"
                )

            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.text_encoder = torch.compile(pipe.text_encoder, mode="reduce-overhead", fullgraph=True)

            if self.has_two_text_encoders:
                pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2, mode="reduce-overhead", fullgraph=True)

            # Just makes sure it works..
            _ = pipe(**inputs, generator=torch.manual_seed(0)).images

    def test_modify_padding_mode(self):

        for scheduler_cls in [DDIMScheduler, LCMScheduler]:
            components, _, _ = self.get_dummy_components(scheduler_cls)
            pipe = self.pipeline_class(**components)
            pipe = pipe.to(torch_device)
            pipe.set_progress_bar_config(disable=None)
            _pad_mode = "circular"
            set_pad_mode(pipe.vae, _pad_mode)
            set_pad_mode(pipe.unet, _pad_mode)

            _, _, inputs = self.get_dummy_inputs()
            _ = pipe(**inputs).images