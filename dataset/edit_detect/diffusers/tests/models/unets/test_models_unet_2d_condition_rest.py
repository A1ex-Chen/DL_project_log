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

import copy
import gc
import os
import tempfile
import unittest
from collections import OrderedDict

import torch
from parameterized import parameterized
from pytest import mark

from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import (
    CustomDiffusionAttnProcessor,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from diffusers.models.embeddings import ImageProjection, IPAdapterFaceIDImageProjection, IPAdapterPlusImageProjection
from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    floats_tensor,
    load_hf_numpy,
    require_torch_accelerator,
    require_torch_accelerator_with_fp16,
    require_torch_accelerator_with_training,
    require_torch_gpu,
    skip_mps,
    slow,
    torch_all_close,
    torch_device,
)

from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


logger = logging.get_logger(__name__)

enable_full_determinism()










class UNet2DConditionModelTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = UNet2DConditionModel
    main_input_name = "sample"
    # We override the items here because the unet under consideration is small.
    model_split_percents = [0.5, 0.3, 0.4]

    @property

    @property

    @property


    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )

    @require_torch_accelerator_with_training










    @parameterized.expand(
        [
            # fmt: off
            [torch.bool],
            [torch.long],
            [torch.float],
            # fmt: on
        ]
    )

    # see diffusers.models.attention_processor::Attention#prepare_attention_mask
    # note: we may not need to fix mask padding to work for stable-diffusion cross-attn masks.
    # since the use-case (somebody passes in a too-short cross-attn mask) is pretty esoteric.
    # maybe it's fine that this only works for the unclip use-case.
    @mark.skip(
        reason="we currently pad mask by target_length tokens (what unclip needs), whereas stable-diffusion's cross-attn needs to instead pad by remaining_length."
    )



    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )






@slow
class UNet2DConditionModelIntegrationTests(unittest.TestCase):




    @require_torch_gpu

    @require_torch_gpu

    @require_torch_gpu

    @require_torch_gpu


    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4424, 0.1510, -0.1937, 0.2118, 0.3746, -0.3957, 0.0160, -0.0435]],
            [47, 0.55, [-0.1508, 0.0379, -0.3075, 0.2540, 0.3633, -0.0821, 0.1719, -0.0207]],
            [21, 0.89, [-0.6479, 0.6364, -0.3464, 0.8697, 0.4443, -0.6289, -0.0091, 0.1778]],
            [9, 1000, [0.8888, -0.5659, 0.5834, -0.7469, 1.1912, -0.3923, 1.1241, -0.4424]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2323, -0.1304, 0.0813, -0.3093, -0.0919, -0.1571, -0.1125, -0.5806]],
            [17, 0.55, [-0.0831, -0.2443, 0.0901, -0.0919, 0.3396, 0.0103, -0.3743, 0.0701]],
            [8, 0.89, [-0.4863, 0.0859, 0.0875, -0.1658, 0.9199, -0.0114, 0.4839, 0.4639]],
            [3, 1000, [-0.5649, 0.2402, -0.5518, 0.1248, 1.1328, -0.2443, -0.0325, -1.0078]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4430, 0.1570, -0.1867, 0.2376, 0.3205, -0.3681, 0.0525, -0.0722]],
            [47, 0.55, [-0.1415, 0.0129, -0.3136, 0.2257, 0.3430, -0.0536, 0.2114, -0.0436]],
            [21, 0.89, [-0.7091, 0.6664, -0.3643, 0.9032, 0.4499, -0.6541, 0.0139, 0.1750]],
            [9, 1000, [0.8878, -0.5659, 0.5844, -0.7442, 1.1883, -0.3927, 1.1192, -0.4423]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2695, -0.1669, 0.0073, -0.3181, -0.1187, -0.1676, -0.1395, -0.5972]],
            [17, 0.55, [-0.1290, -0.2588, 0.0551, -0.0916, 0.3286, 0.0238, -0.3669, 0.0322]],
            [8, 0.89, [-0.5283, 0.1198, 0.0870, -0.1141, 0.9189, -0.0150, 0.5474, 0.4319]],
            [3, 1000, [-0.5601, 0.2411, -0.5435, 0.1268, 1.1338, -0.2427, -0.0280, -1.0020]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.7639, 0.0106, -0.1615, -0.3487, -0.0423, -0.7972, 0.0085, -0.4858]],
            [47, 0.55, [-0.6564, 0.0795, -1.9026, -0.6258, 1.8235, 1.2056, 1.2169, 0.9073]],
            [21, 0.89, [0.0327, 0.4399, -0.6358, 0.3417, 0.4120, -0.5621, -0.0397, -1.0430]],
            [9, 1000, [0.1600, 0.7303, -1.0556, -0.3515, -0.7440, -1.2037, -1.8149, -1.8931]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.1047, -1.7227, 0.1067, 0.0164, -0.5698, -0.4172, -0.1388, 1.1387]],
            [17, 0.55, [0.0975, -0.2856, -0.3508, -0.4600, 0.3376, 0.2930, -0.2747, -0.7026]],
            [8, 0.89, [-0.0952, 0.0183, -0.5825, -0.1981, 0.1131, 0.4668, -0.0395, -0.3486]],
            [3, 1000, [0.4790, 0.4949, -1.0732, -0.7158, 0.7959, -0.9478, 0.1105, -0.9741]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [0.1514, 0.0807, 0.1624, 0.1016, -0.1896, 0.0263, 0.0677, 0.2310]],
            [17, 0.55, [0.1164, -0.0216, 0.0170, 0.1589, -0.3120, 0.1005, -0.0581, -0.1458]],
            [8, 0.89, [-0.1758, -0.0169, 0.1004, -0.1411, 0.1312, 0.1103, -0.1996, 0.2139]],
            [3, 1000, [0.1214, 0.0352, -0.0731, -0.1562, -0.0994, -0.0906, -0.2340, -0.0539]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16

        # retrieve number of attention layers
        for module in model.children():
            check_sliceable_dim_attr(module)

    def test_gradient_checkpointing_is_applied(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model_class_copy = copy.copy(self.model_class)

        modules_with_gc_enabled = {}

        # now monkey patch the following function:
        #     def _set_gradient_checkpointing(self, module, value=False):
        #         if hasattr(module, "gradient_checkpointing"):
        #             module.gradient_checkpointing = value


        model_class_copy._set_gradient_checkpointing = _set_gradient_checkpointing_new

        model = model_class_copy(**init_dict)
        model.enable_gradient_checkpointing()

        EXPECTED_SET = {
            "CrossAttnUpBlock2D",
            "CrossAttnDownBlock2D",
            "UNetMidBlock2DCrossAttn",
            "UpBlock2D",
            "Transformer2DModel",
            "DownBlock2D",
        }

        assert set(modules_with_gc_enabled.keys()) == EXPECTED_SET
        assert all(modules_with_gc_enabled.values()), "All modules should be enabled"

    def test_special_attn_proc(self):
        class AttnEasyProc(torch.nn.Module):


        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        processor = AttnEasyProc(5.0)

        model.set_attn_processor(processor)
        model(**inputs_dict, cross_attention_kwargs={"number": 123}).sample

        assert processor.counter == 8
        assert processor.is_run
        assert processor.number == 123

    @parameterized.expand(
        [
            # fmt: off
            [torch.bool],
            [torch.long],
            [torch.float],
            # fmt: on
        ]
    )
    def test_model_xattn_mask(self, mask_dtype):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**{**init_dict, "attention_head_dim": (8, 16), "block_out_channels": (16, 32)})
        model.to(torch_device)
        model.eval()

        cond = inputs_dict["encoder_hidden_states"]
        with torch.no_grad():
            full_cond_out = model(**inputs_dict).sample
            assert full_cond_out is not None

            keepall_mask = torch.ones(*cond.shape[:-1], device=cond.device, dtype=mask_dtype)
            full_cond_keepallmask_out = model(**{**inputs_dict, "encoder_attention_mask": keepall_mask}).sample
            assert full_cond_keepallmask_out.allclose(
                full_cond_out, rtol=1e-05, atol=1e-05
            ), "a 'keep all' mask should give the same result as no mask"

            trunc_cond = cond[:, :-1, :]
            trunc_cond_out = model(**{**inputs_dict, "encoder_hidden_states": trunc_cond}).sample
            assert not trunc_cond_out.allclose(
                full_cond_out, rtol=1e-05, atol=1e-05
            ), "discarding the last token from our cond should change the result"

            batch, tokens, _ = cond.shape
            mask_last = (torch.arange(tokens) < tokens - 1).expand(batch, -1).to(cond.device, mask_dtype)
            masked_cond_out = model(**{**inputs_dict, "encoder_attention_mask": mask_last}).sample
            assert masked_cond_out.allclose(
                trunc_cond_out, rtol=1e-05, atol=1e-05
            ), "masking the last token from our cond should be equivalent to truncating that token out of the condition"

    # see diffusers.models.attention_processor::Attention#prepare_attention_mask
    # note: we may not need to fix mask padding to work for stable-diffusion cross-attn masks.
    # since the use-case (somebody passes in a too-short cross-attn mask) is pretty esoteric.
    # maybe it's fine that this only works for the unclip use-case.
    @mark.skip(
        reason="we currently pad mask by target_length tokens (what unclip needs), whereas stable-diffusion's cross-attn needs to instead pad by remaining_length."
    )
    def test_model_xattn_padding(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        model = self.model_class(**{**init_dict, "attention_head_dim": (8, 16)})
        model.to(torch_device)
        model.eval()

        cond = inputs_dict["encoder_hidden_states"]
        with torch.no_grad():
            full_cond_out = model(**inputs_dict).sample
            assert full_cond_out is not None

            batch, tokens, _ = cond.shape
            keeplast_mask = (torch.arange(tokens) == tokens - 1).expand(batch, -1).to(cond.device, torch.bool)
            keeplast_out = model(**{**inputs_dict, "encoder_attention_mask": keeplast_mask}).sample
            assert not keeplast_out.allclose(full_cond_out), "a 'keep last token' mask should change the result"

            trunc_mask = torch.zeros(batch, tokens - 1, device=cond.device, dtype=torch.bool)
            trunc_mask_out = model(**{**inputs_dict, "encoder_attention_mask": trunc_mask}).sample
            assert trunc_mask_out.allclose(
                keeplast_out
            ), "a mask with fewer tokens than condition, will be padded with 'keep' tokens. a 'discard-all' mask missing the final token is thus equivalent to a 'keep last' mask."

    def test_custom_diffusion_processors(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        custom_diffusion_attn_procs = create_custom_diffusion_layers(model, mock_weights=False)

        # make sure we can set a list of attention processors
        model.set_attn_processor(custom_diffusion_attn_procs)
        model.to(torch_device)

        # test that attn processors can be set to itself
        model.set_attn_processor(model.attn_processors)

        with torch.no_grad():
            sample2 = model(**inputs_dict).sample

        assert (sample1 - sample2).abs().max() < 3e-3

    def test_custom_diffusion_save_load(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            old_sample = model(**inputs_dict).sample

        custom_diffusion_attn_procs = create_custom_diffusion_layers(model, mock_weights=False)
        model.set_attn_processor(custom_diffusion_attn_procs)

        with torch.no_grad():
            sample = model(**inputs_dict).sample

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_attn_procs(tmpdirname, safe_serialization=False)
            self.assertTrue(os.path.isfile(os.path.join(tmpdirname, "pytorch_custom_diffusion_weights.bin")))
            torch.manual_seed(0)
            new_model = self.model_class(**init_dict)
            new_model.load_attn_procs(tmpdirname, weight_name="pytorch_custom_diffusion_weights.bin")
            new_model.to(torch_device)

        with torch.no_grad():
            new_sample = new_model(**inputs_dict).sample

        assert (sample - new_sample).abs().max() < 1e-4

        # custom diffusion and no custom diffusion should be the same
        assert (sample - old_sample).abs().max() < 3e-3

    @unittest.skipIf(
        torch_device != "cuda" or not is_xformers_available(),
        reason="XFormers attention is only available with CUDA and `xformers` installed",
    )
    def test_custom_diffusion_xformers_on_off(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)
        custom_diffusion_attn_procs = create_custom_diffusion_layers(model, mock_weights=False)
        model.set_attn_processor(custom_diffusion_attn_procs)

        # default
        with torch.no_grad():
            sample = model(**inputs_dict).sample

            model.enable_xformers_memory_efficient_attention()
            on_sample = model(**inputs_dict).sample

            model.disable_xformers_memory_efficient_attention()
            off_sample = model(**inputs_dict).sample

        assert (sample - on_sample).abs().max() < 1e-4
        assert (sample - off_sample).abs().max() < 1e-4

    def test_pickle(self):
        # enable deterministic behavior for gradient checkpointing
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        with torch.no_grad():
            sample = model(**inputs_dict).sample

        sample_copy = copy.copy(sample)

        assert (sample - sample_copy).abs().max() < 1e-4

    def test_asymmetrical_unet(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()
        # Add asymmetry to configs
        init_dict["transformer_layers_per_block"] = [[3, 2], 1]
        init_dict["reverse_transformer_layers_per_block"] = [[3, 4], 1]

        torch.manual_seed(0)
        model = self.model_class(**init_dict)
        model.to(torch_device)

        output = model(**inputs_dict).sample
        expected_shape = inputs_dict["sample"].shape

        # Check if input and output shapes are the same
        self.assertEqual(output.shape, expected_shape, "Input and output shapes do not match")

    def test_ip_adapter(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        # forward pass without ip-adapter
        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        # update inputs_dict for ip-adapter
        batch_size = inputs_dict["encoder_hidden_states"].shape[0]
        # for ip-adapter image_embeds has shape [batch_size, num_image, embed_dim]
        image_embeds = floats_tensor((batch_size, 1, model.config.cross_attention_dim)).to(torch_device)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds]}

        # make ip_adapter_1 and ip_adapter_2
        ip_adapter_1 = create_ip_adapter_state_dict(model)

        image_proj_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["image_proj"].items()}
        cross_attn_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["ip_adapter"].items()}
        ip_adapter_2 = {}
        ip_adapter_2.update({"image_proj": image_proj_state_dict_2, "ip_adapter": cross_attn_state_dict_2})

        # forward pass ip_adapter_1
        model._load_ip_adapter_weights([ip_adapter_1])
        assert model.config.encoder_hid_dim_type == "ip_image_proj"
        assert model.encoder_hid_proj is not None
        assert model.down_blocks[0].attentions[0].transformer_blocks[0].attn2.processor.__class__.__name__ in (
            "IPAdapterAttnProcessor",
            "IPAdapterAttnProcessor2_0",
        )
        with torch.no_grad():
            sample2 = model(**inputs_dict).sample

        # forward pass with ip_adapter_2
        model._load_ip_adapter_weights([ip_adapter_2])
        with torch.no_grad():
            sample3 = model(**inputs_dict).sample

        # forward pass with ip_adapter_1 again
        model._load_ip_adapter_weights([ip_adapter_1])
        with torch.no_grad():
            sample4 = model(**inputs_dict).sample

        # forward pass with multiple ip-adapters and multiple images
        model._load_ip_adapter_weights([ip_adapter_1, ip_adapter_2])
        # set the scale for ip_adapter_2 to 0 so that result should be same as only load ip_adapter_1
        for attn_processor in model.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                attn_processor.scale = [1, 0]
        image_embeds_multi = image_embeds.repeat(1, 2, 1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds_multi, image_embeds_multi]}
        with torch.no_grad():
            sample5 = model(**inputs_dict).sample

        # forward pass with single ip-adapter & single image when image_embeds is not a list and a 2-d tensor
        image_embeds = image_embeds.squeeze(1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": image_embeds}

        model._load_ip_adapter_weights(ip_adapter_1)
        with torch.no_grad():
            sample6 = model(**inputs_dict).sample

        assert not sample1.allclose(sample2, atol=1e-4, rtol=1e-4)
        assert not sample2.allclose(sample3, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample4, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample5, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample6, atol=1e-4, rtol=1e-4)

    def test_ip_adapter_plus(self):
        init_dict, inputs_dict = self.prepare_init_args_and_inputs_for_common()

        init_dict["block_out_channels"] = (16, 32)
        init_dict["attention_head_dim"] = (8, 16)

        model = self.model_class(**init_dict)
        model.to(torch_device)

        # forward pass without ip-adapter
        with torch.no_grad():
            sample1 = model(**inputs_dict).sample

        # update inputs_dict for ip-adapter
        batch_size = inputs_dict["encoder_hidden_states"].shape[0]
        # for ip-adapter-plus image_embeds has shape [batch_size, num_image, sequence_length, embed_dim]
        image_embeds = floats_tensor((batch_size, 1, 1, model.config.cross_attention_dim)).to(torch_device)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds]}

        # make ip_adapter_1 and ip_adapter_2
        ip_adapter_1 = create_ip_adapter_plus_state_dict(model)

        image_proj_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["image_proj"].items()}
        cross_attn_state_dict_2 = {k: w + 1.0 for k, w in ip_adapter_1["ip_adapter"].items()}
        ip_adapter_2 = {}
        ip_adapter_2.update({"image_proj": image_proj_state_dict_2, "ip_adapter": cross_attn_state_dict_2})

        # forward pass ip_adapter_1
        model._load_ip_adapter_weights([ip_adapter_1])
        assert model.config.encoder_hid_dim_type == "ip_image_proj"
        assert model.encoder_hid_proj is not None
        assert model.down_blocks[0].attentions[0].transformer_blocks[0].attn2.processor.__class__.__name__ in (
            "IPAdapterAttnProcessor",
            "IPAdapterAttnProcessor2_0",
        )
        with torch.no_grad():
            sample2 = model(**inputs_dict).sample

        # forward pass with ip_adapter_2
        model._load_ip_adapter_weights([ip_adapter_2])
        with torch.no_grad():
            sample3 = model(**inputs_dict).sample

        # forward pass with ip_adapter_1 again
        model._load_ip_adapter_weights([ip_adapter_1])
        with torch.no_grad():
            sample4 = model(**inputs_dict).sample

        # forward pass with multiple ip-adapters and multiple images
        model._load_ip_adapter_weights([ip_adapter_1, ip_adapter_2])
        # set the scale for ip_adapter_2 to 0 so that result should be same as only load ip_adapter_1
        for attn_processor in model.attn_processors.values():
            if isinstance(attn_processor, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0)):
                attn_processor.scale = [1, 0]
        image_embeds_multi = image_embeds.repeat(1, 2, 1, 1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": [image_embeds_multi, image_embeds_multi]}
        with torch.no_grad():
            sample5 = model(**inputs_dict).sample

        # forward pass with single ip-adapter & single image when image_embeds is a 3-d tensor
        image_embeds = image_embeds[:,].squeeze(1)
        inputs_dict["added_cond_kwargs"] = {"image_embeds": image_embeds}

        model._load_ip_adapter_weights(ip_adapter_1)
        with torch.no_grad():
            sample6 = model(**inputs_dict).sample

        assert not sample1.allclose(sample2, atol=1e-4, rtol=1e-4)
        assert not sample2.allclose(sample3, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample4, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample5, atol=1e-4, rtol=1e-4)
        assert sample2.allclose(sample6, atol=1e-4, rtol=1e-4)


@slow
class UNet2DConditionModelIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_latents(self, seed=0, shape=(4, 4, 64, 64), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_unet_model(self, fp16=False, model_id="CompVis/stable-diffusion-v1-4"):
        revision = "fp16" if fp16 else None
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch_dtype, revision=revision
        )
        model.to(torch_device).eval()

        return model

    @require_torch_gpu
    def test_set_attention_slice_auto(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        unet = self.get_unet_model()
        unet.set_attention_slice("auto")

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    @require_torch_gpu
    def test_set_attention_slice_max(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        unet = self.get_unet_model()
        unet.set_attention_slice("max")

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    @require_torch_gpu
    def test_set_attention_slice_int(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        unet = self.get_unet_model()
        unet.set_attention_slice(2)

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    @require_torch_gpu
    def test_set_attention_slice_list(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # there are 32 sliceable layers
        slice_list = 16 * [2, 3]
        unet = self.get_unet_model()
        unet.set_attention_slice(slice_list)

        latents = self.get_latents(33)
        encoder_hidden_states = self.get_encoder_hidden_states(33)
        timestep = 1

        with torch.no_grad():
            _ = unet(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        mem_bytes = torch.cuda.max_memory_allocated()

        assert mem_bytes < 5 * 10**9

    def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        hidden_states = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return hidden_states

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4424, 0.1510, -0.1937, 0.2118, 0.3746, -0.3957, 0.0160, -0.0435]],
            [47, 0.55, [-0.1508, 0.0379, -0.3075, 0.2540, 0.3633, -0.0821, 0.1719, -0.0207]],
            [21, 0.89, [-0.6479, 0.6364, -0.3464, 0.8697, 0.4443, -0.6289, -0.0091, 0.1778]],
            [9, 1000, [0.8888, -0.5659, 0.5834, -0.7469, 1.1912, -0.3923, 1.1241, -0.4424]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_v1_4(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2323, -0.1304, 0.0813, -0.3093, -0.0919, -0.1571, -0.1125, -0.5806]],
            [17, 0.55, [-0.0831, -0.2443, 0.0901, -0.0919, 0.3396, 0.0103, -0.3743, 0.0701]],
            [8, 0.89, [-0.4863, 0.0859, 0.0875, -0.1658, 0.9199, -0.0114, 0.4839, 0.4639]],
            [3, 1000, [-0.5649, 0.2402, -0.5518, 0.1248, 1.1328, -0.2443, -0.0325, -1.0078]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_v1_4_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="CompVis/stable-diffusion-v1-4", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.4430, 0.1570, -0.1867, 0.2376, 0.3205, -0.3681, 0.0525, -0.0722]],
            [47, 0.55, [-0.1415, 0.0129, -0.3136, 0.2257, 0.3430, -0.0536, 0.2114, -0.0436]],
            [21, 0.89, [-0.7091, 0.6664, -0.3643, 0.9032, 0.4499, -0.6541, 0.0139, 0.1750]],
            [9, 1000, [0.8878, -0.5659, 0.5844, -0.7442, 1.1883, -0.3927, 1.1192, -0.4423]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps
    def test_compvis_sd_v1_5(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-v1-5")
        latents = self.get_latents(seed)
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.2695, -0.1669, 0.0073, -0.3181, -0.1187, -0.1676, -0.1395, -0.5972]],
            [17, 0.55, [-0.1290, -0.2588, 0.0551, -0.0916, 0.3286, 0.0238, -0.3669, 0.0322]],
            [8, 0.89, [-0.5283, 0.1198, 0.0870, -0.1141, 0.9189, -0.0150, 0.5474, 0.4319]],
            [3, 1000, [-0.5601, 0.2411, -0.5435, 0.1268, 1.1338, -0.2427, -0.0280, -1.0020]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_v1_5_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-v1-5", fp16=True)
        latents = self.get_latents(seed, fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, 4, [-0.7639, 0.0106, -0.1615, -0.3487, -0.0423, -0.7972, 0.0085, -0.4858]],
            [47, 0.55, [-0.6564, 0.0795, -1.9026, -0.6258, 1.8235, 1.2056, 1.2169, 0.9073]],
            [21, 0.89, [0.0327, 0.4399, -0.6358, 0.3417, 0.4120, -0.5621, -0.0397, -1.0430]],
            [9, 1000, [0.1600, 0.7303, -1.0556, -0.3515, -0.7440, -1.2037, -1.8149, -1.8931]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps
    def test_compvis_sd_inpaint(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-inpainting")
        latents = self.get_latents(seed, shape=(4, 9, 64, 64))
        encoder_hidden_states = self.get_encoder_hidden_states(seed)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == (4, 4, 64, 64)

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [-0.1047, -1.7227, 0.1067, 0.0164, -0.5698, -0.4172, -0.1388, 1.1387]],
            [17, 0.55, [0.0975, -0.2856, -0.3508, -0.4600, 0.3376, 0.2930, -0.2747, -0.7026]],
            [8, 0.89, [-0.0952, 0.0183, -0.5825, -0.1981, 0.1131, 0.4668, -0.0395, -0.3486]],
            [3, 1000, [0.4790, 0.4949, -1.0732, -0.7158, 0.7959, -0.9478, 0.1105, -0.9741]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_compvis_sd_inpaint_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="runwayml/stable-diffusion-inpainting", fp16=True)
        latents = self.get_latents(seed, shape=(4, 9, 64, 64), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == (4, 4, 64, 64)

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [83, 4, [0.1514, 0.0807, 0.1624, 0.1016, -0.1896, 0.0263, 0.0677, 0.2310]],
            [17, 0.55, [0.1164, -0.0216, 0.0170, 0.1589, -0.3120, 0.1005, -0.0581, -0.1458]],
            [8, 0.89, [-0.1758, -0.0169, 0.1004, -0.1411, 0.1312, 0.1103, -0.1996, 0.2139]],
            [3, 1000, [0.1214, 0.0352, -0.0731, -0.1562, -0.0994, -0.0906, -0.2340, -0.0539]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_stabilityai_sd_v2_fp16(self, seed, timestep, expected_slice):
        model = self.get_unet_model(model_id="stabilityai/stable-diffusion-2", fp16=True)
        latents = self.get_latents(seed, shape=(4, 4, 96, 96), fp16=True)
        encoder_hidden_states = self.get_encoder_hidden_states(seed, shape=(4, 77, 1024), fp16=True)

        timestep = torch.tensor([timestep], dtype=torch.long, device=torch_device)

        with torch.no_grad():
            sample = model(latents, timestep=timestep, encoder_hidden_states=encoder_hidden_states).sample

        assert sample.shape == latents.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)