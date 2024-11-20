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

import gc
import json
import os
import random
import shutil
import sys
import tempfile
import traceback
import unittest
import unittest.mock as mock

import numpy as np
import PIL.Image
import requests_mock
import safetensors.torch
import torch
from parameterized import parameterized
from PIL import Image
from requests.exceptions import HTTPError
from transformers import CLIPImageProcessor, CLIPModel, CLIPTextConfig, CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    ConfigMixin,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    ModelMixin,
    PNDMScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    UNet2DModel,
    UniPCMultistepScheduler,
    logging,
)
from diffusers.pipelines.pipeline_utils import _get_pipeline_class
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
)
from diffusers.utils.testing_utils import (
    CaptureLogger,
    enable_full_determinism,
    floats_tensor,
    get_python_version,
    get_tests_dir,
    load_numpy,
    nightly,
    require_compel,
    require_flax,
    require_onnxruntime,
    require_python39_or_higher,
    require_torch_2,
    require_torch_gpu,
    run_test_in_subprocess,
    slow,
    torch_device,
)
from diffusers.utils.torch_utils import is_compiled_module


enable_full_determinism()


# Will be run via run_test_in_subprocess


class CustomEncoder(ModelMixin, ConfigMixin):


class CustomPipeline(DiffusionPipeline):


class DownloadTests(unittest.TestCase):










    @require_onnxruntime









                # only unet has "no_ema" variant







class CustomPipelineTests(unittest.TestCase):








    @slow
    @require_torch_gpu



class PipelineFastTests(unittest.TestCase):





    @property

    @property

    @property

    @parameterized.expand(
        [
            [DDIMScheduler, DDIMPipeline, 32],
            [DDPMScheduler, DDPMPipeline, 32],
            [DDIMScheduler, DDIMPipeline, (32, 64)],
            [DDPMScheduler, DDPMPipeline, (64, 32)],
        ]
    )


    @require_torch_gpu














@slow
@require_torch_gpu
class PipelineSlowTests(unittest.TestCase):





    @require_python39_or_higher
    @require_torch_2
    @unittest.skipIf(
        get_python_version == (3, 12),
        reason="Torch Dynamo isn't yet supported for Python 3.12.",
    )




    @require_flax

    @require_compel


@nightly
@require_torch_gpu
class PipelineNightlyTests(unittest.TestCase):




            return Out()

        return extract

    @parameterized.expand(
        [
            [DDIMScheduler, DDIMPipeline, 32],
            [DDPMScheduler, DDPMPipeline, 32],
            [DDIMScheduler, DDIMPipeline, (32, 64)],
            [DDPMScheduler, DDPMPipeline, (64, 32)],
        ]
    )
    def test_uncond_unet_components(self, scheduler_fn=DDPMScheduler, pipeline_fn=DDPMPipeline, sample_size=32):
        unet = self.dummy_uncond_unet(sample_size)
        scheduler = scheduler_fn()
        pipeline = pipeline_fn(unet, scheduler).to(torch_device)

        generator = torch.manual_seed(0)
        out_image = pipeline(
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images
        sample_size = (sample_size, sample_size) if isinstance(sample_size, int) else sample_size
        assert out_image.shape == (1, *sample_size, 3)

    def test_stable_diffusion_components(self):
        """Test that components property works correctly"""
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        image = self.dummy_image().cpu().permute(0, 2, 3, 1)[0]
        init_image = Image.fromarray(np.uint8(image)).convert("RGB")
        mask_image = Image.fromarray(np.uint8(image + 4)).convert("RGB").resize((32, 32))

        # make sure here that pndm scheduler skips prk
        inpaint = StableDiffusionInpaintPipelineLegacy(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        ).to(torch_device)
        img2img = StableDiffusionImg2ImgPipeline(**inpaint.components, image_encoder=None).to(torch_device)
        text2img = StableDiffusionPipeline(**inpaint.components, image_encoder=None).to(torch_device)

        prompt = "A painting of a squirrel eating a burger"

        generator = torch.manual_seed(0)
        image_inpaint = inpaint(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
            mask_image=mask_image,
        ).images
        image_img2img = img2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
            image=init_image,
        ).images
        image_text2img = text2img(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images

        assert image_inpaint.shape == (1, 32, 32, 3)
        assert image_img2img.shape == (1, 32, 32, 3)
        assert image_text2img.shape == (1, 64, 64, 3)

    @require_torch_gpu
    def test_pipe_false_offload_warn(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        sd.enable_model_cpu_offload()

        logger = logging.get_logger("diffusers.pipelines.pipeline_utils")
        with CaptureLogger(logger) as cap_logger:
            sd.to("cuda")

        assert "It is strongly recommended against doing so" in str(cap_logger)

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

    def test_set_scheduler(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DDIMScheduler)
        sd.scheduler = DDPMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DDPMScheduler)
        sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, PNDMScheduler)
        sd.scheduler = LMSDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, LMSDiscreteScheduler)
        sd.scheduler = EulerDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, EulerDiscreteScheduler)
        sd.scheduler = EulerAncestralDiscreteScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, EulerAncestralDiscreteScheduler)
        sd.scheduler = DPMSolverMultistepScheduler.from_config(sd.scheduler.config)
        assert isinstance(sd.scheduler, DPMSolverMultistepScheduler)

    def test_set_component_to_none(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        pipeline = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        generator = torch.Generator(device="cpu").manual_seed(0)

        prompt = "This is a flower"

        out_image = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=1,
            output_type="np",
        ).images

        pipeline.feature_extractor = None
        generator = torch.Generator(device="cpu").manual_seed(0)
        out_image_2 = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=1,
            output_type="np",
        ).images

        assert out_image.shape == (1, 64, 64, 3)
        assert np.abs(out_image - out_image_2).max() < 1e-3

    def test_optional_components_is_none(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        items = {
            "feature_extractor": self.dummy_extractor,
            "unet": unet,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder": bert,
            "tokenizer": tokenizer,
            "safety_checker": None,
            # we don't add an image encoder
        }

        pipeline = StableDiffusionPipeline(**items)

        assert sorted(pipeline.components.keys()) == sorted(["image_encoder"] + list(items.keys()))
        assert pipeline.image_encoder is None

    def test_set_scheduler_consistency(self):
        unet = self.dummy_cond_unet()
        pndm = PNDMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        ddim = DDIMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=pndm,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        pndm_config = sd.scheduler.config
        sd.scheduler = DDPMScheduler.from_config(pndm_config)
        sd.scheduler = PNDMScheduler.from_config(sd.scheduler.config)
        pndm_config_2 = sd.scheduler.config
        pndm_config_2 = {k: v for k, v in pndm_config_2.items() if k in pndm_config}

        assert dict(pndm_config) == dict(pndm_config_2)

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=ddim,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        ddim_config = sd.scheduler.config
        sd.scheduler = LMSDiscreteScheduler.from_config(ddim_config)
        sd.scheduler = DDIMScheduler.from_config(sd.scheduler.config)
        ddim_config_2 = sd.scheduler.config
        ddim_config_2 = {k: v for k, v in ddim_config_2.items() if k in ddim_config}

        assert dict(ddim_config) == dict(ddim_config_2)

    def test_save_safe_serialization(self):
        pipeline = StableDiffusionPipeline.from_pretrained("hf-internal-testing/tiny-stable-diffusion-torch")
        with tempfile.TemporaryDirectory() as tmpdirname:
            pipeline.save_pretrained(tmpdirname, safe_serialization=True)

            # Validate that the VAE safetensor exists and are of the correct format
            vae_path = os.path.join(tmpdirname, "vae", "diffusion_pytorch_model.safetensors")
            assert os.path.exists(vae_path), f"Could not find {vae_path}"
            _ = safetensors.torch.load_file(vae_path)

            # Validate that the UNet safetensor exists and are of the correct format
            unet_path = os.path.join(tmpdirname, "unet", "diffusion_pytorch_model.safetensors")
            assert os.path.exists(unet_path), f"Could not find {unet_path}"
            _ = safetensors.torch.load_file(unet_path)

            # Validate that the text encoder safetensor exists and are of the correct format
            text_encoder_path = os.path.join(tmpdirname, "text_encoder", "model.safetensors")
            assert os.path.exists(text_encoder_path), f"Could not find {text_encoder_path}"
            _ = safetensors.torch.load_file(text_encoder_path)

            pipeline = StableDiffusionPipeline.from_pretrained(tmpdirname)
            assert pipeline.unet is not None
            assert pipeline.vae is not None
            assert pipeline.text_encoder is not None
            assert pipeline.scheduler is not None
            assert pipeline.feature_extractor is not None

    def test_no_pytorch_download_when_doing_safetensors(self):
        # by default we don't download
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all", cache_dir=tmpdirname
            )

            path = os.path.join(
                tmpdirname,
                "models--hf-internal-testing--diffusers-stable-diffusion-tiny-all",
                "snapshots",
                "07838d72e12f9bcec1375b0482b80c1d399be843",
                "unet",
            )
            # safetensors exists
            assert os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            # pytorch does not
            assert not os.path.exists(os.path.join(path, "diffusion_pytorch_model.bin"))

    def test_no_safetensors_download_when_doing_pytorch(self):
        use_safetensors = False

        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = StableDiffusionPipeline.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all",
                cache_dir=tmpdirname,
                use_safetensors=use_safetensors,
            )

            path = os.path.join(
                tmpdirname,
                "models--hf-internal-testing--diffusers-stable-diffusion-tiny-all",
                "snapshots",
                "07838d72e12f9bcec1375b0482b80c1d399be843",
                "unet",
            )
            # safetensors does not exists
            assert not os.path.exists(os.path.join(path, "diffusion_pytorch_model.safetensors"))
            # pytorch does
            assert os.path.exists(os.path.join(path, "diffusion_pytorch_model.bin"))

    def test_optional_components(self):
        unet = self.dummy_cond_unet()
        pndm = PNDMScheduler.from_config("hf-internal-testing/tiny-stable-diffusion-torch", subfolder="scheduler")
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        orig_sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=pndm,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=unet,
            feature_extractor=self.dummy_extractor,
        )
        sd = orig_sd

        assert sd.config.requires_safety_checker is True

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)

            # Test that passing None works
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname, feature_extractor=None, safety_checker=None, requires_safety_checker=False
            )

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)

            # Test that loading previous None works
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            orig_sd.save_pretrained(tmpdirname)

            # Test that loading without any directory works
            shutil.rmtree(os.path.join(tmpdirname, "safety_checker"))
            with open(os.path.join(tmpdirname, sd.config_name)) as f:
                config = json.load(f)
                config["safety_checker"] = [None, None]
            with open(os.path.join(tmpdirname, sd.config_name), "w") as f:
                json.dump(config, f)

            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, requires_safety_checker=False)
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

            # Test that loading from deleted model index works
            with open(os.path.join(tmpdirname, sd.config_name)) as f:
                config = json.load(f)
                del config["safety_checker"]
                del config["feature_extractor"]
            with open(os.path.join(tmpdirname, sd.config_name), "w") as f:
                json.dump(config, f)

            sd = StableDiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor == (None, None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)

            # Test that partially loading works
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)

            assert sd.config.requires_safety_checker is False
            assert sd.config.safety_checker == (None, None)
            assert sd.config.feature_extractor != (None, None)

            # Test that partially loading works
            sd = StableDiffusionPipeline.from_pretrained(
                tmpdirname,
                feature_extractor=self.dummy_extractor,
                safety_checker=unet,
                requires_safety_checker=[True, True],
            )

            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)
            sd = StableDiffusionPipeline.from_pretrained(tmpdirname, feature_extractor=self.dummy_extractor)

            assert sd.config.requires_safety_checker == [True, True]
            assert sd.config.safety_checker != (None, None)
            assert sd.config.feature_extractor != (None, None)

    def test_name_or_path(self):
        model_path = "hf-internal-testing/tiny-stable-diffusion-torch"
        sd = DiffusionPipeline.from_pretrained(model_path)

        assert sd.name_or_path == model_path

        with tempfile.TemporaryDirectory() as tmpdirname:
            sd.save_pretrained(tmpdirname)
            sd = DiffusionPipeline.from_pretrained(tmpdirname)

            assert sd.name_or_path == tmpdirname

    def test_error_no_variant_available(self):
        variant = "fp16"
        with self.assertRaises(ValueError) as error_context:
            _ = StableDiffusionPipeline.download(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all", variant=variant
            )

        assert "but no such modeling files are available" in str(error_context.exception)
        assert variant in str(error_context.exception)

    def test_pipe_to(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        device_type = torch.device(torch_device).type

        sd1 = sd.to(device_type)
        sd2 = sd.to(torch.device(device_type))
        sd3 = sd.to(device_type, torch.float32)
        sd4 = sd.to(device=device_type)
        sd5 = sd.to(torch_device=device_type)
        sd6 = sd.to(device_type, dtype=torch.float32)
        sd7 = sd.to(device_type, torch_dtype=torch.float32)

        assert sd1.device.type == device_type
        assert sd2.device.type == device_type
        assert sd3.device.type == device_type
        assert sd4.device.type == device_type
        assert sd5.device.type == device_type
        assert sd6.device.type == device_type
        assert sd7.device.type == device_type

        sd1 = sd.to(torch.float16)
        sd2 = sd.to(None, torch.float16)
        sd3 = sd.to(dtype=torch.float16)
        sd4 = sd.to(dtype=torch.float16)
        sd5 = sd.to(None, dtype=torch.float16)
        sd6 = sd.to(None, torch_dtype=torch.float16)

        assert sd1.dtype == torch.float16
        assert sd2.dtype == torch.float16
        assert sd3.dtype == torch.float16
        assert sd4.dtype == torch.float16
        assert sd5.dtype == torch.float16
        assert sd6.dtype == torch.float16

        sd1 = sd.to(device=device_type, dtype=torch.float16)
        sd2 = sd.to(torch_device=device_type, torch_dtype=torch.float16)
        sd3 = sd.to(device_type, torch.float16)

        assert sd1.dtype == torch.float16
        assert sd2.dtype == torch.float16
        assert sd3.dtype == torch.float16

        assert sd1.device.type == device_type
        assert sd2.device.type == device_type
        assert sd3.device.type == device_type

    def test_pipe_same_device_id_offload(self):
        unet = self.dummy_cond_unet()
        scheduler = PNDMScheduler(skip_prk_steps=True)
        vae = self.dummy_vae
        bert = self.dummy_text_encoder
        tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")

        sd = StableDiffusionPipeline(
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            text_encoder=bert,
            tokenizer=tokenizer,
            safety_checker=None,
            feature_extractor=self.dummy_extractor,
        )

        sd.enable_model_cpu_offload(gpu_id=5)
        assert sd._offload_gpu_id == 5
        sd.maybe_free_model_hooks()
        assert sd._offload_gpu_id == 5


@slow
@require_torch_gpu
class PipelineSlowTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_smart_download(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        with tempfile.TemporaryDirectory() as tmpdirname:
            _ = DiffusionPipeline.from_pretrained(model_id, cache_dir=tmpdirname, force_download=True)
            local_repo_name = "--".join(["models"] + model_id.split("/"))
            snapshot_dir = os.path.join(tmpdirname, local_repo_name, "snapshots")
            snapshot_dir = os.path.join(snapshot_dir, os.listdir(snapshot_dir)[0])

            # inspect all downloaded files to make sure that everything is included
            assert os.path.isfile(os.path.join(snapshot_dir, DiffusionPipeline.config_name))
            assert os.path.isfile(os.path.join(snapshot_dir, CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "scheduler", SCHEDULER_CONFIG_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            assert os.path.isfile(os.path.join(snapshot_dir, "unet", WEIGHTS_NAME))
            # let's make sure the super large numpy file:
            # https://huggingface.co/hf-internal-testing/unet-pipeline-dummy/blob/main/big_array.npy
            # is not downloaded, but all the expected ones
            assert not os.path.isfile(os.path.join(snapshot_dir, "big_array.npy"))

    def test_warning_unused_kwargs(self):
        model_id = "hf-internal-testing/unet-pipeline-dummy"
        logger = logging.get_logger("diffusers.pipelines")
        with tempfile.TemporaryDirectory() as tmpdirname:
            with CaptureLogger(logger) as cap_logger:
                DiffusionPipeline.from_pretrained(
                    model_id,
                    not_used=True,
                    cache_dir=tmpdirname,
                    force_download=True,
                )

        assert (
            cap_logger.out.strip().split("\n")[-1]
            == "Keyword arguments {'not_used': True} are not expected by DDPMPipeline and will be ignored."
        )

    def test_from_save_pretrained(self):
        # 1. Load models
        model = UNet2DModel(
            block_out_channels=(32, 64),
            layers_per_block=2,
            sample_size=32,
            in_channels=3,
            out_channels=3,
            down_block_types=("DownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        )
        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline(model, scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPMPipeline.from_pretrained(tmpdirname)
            new_ddpm.to(torch_device)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        new_image = new_ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        assert np.abs(image - new_image).max() < 1e-5, "Models don't give the same forward pass"

    @require_python39_or_higher
    @require_torch_2
    @unittest.skipIf(
        get_python_version == (3, 12),
        reason="Torch Dynamo isn't yet supported for Python 3.12.",
    )
    def test_from_save_pretrained_dynamo(self):
        run_test_in_subprocess(test_case=self, target_func=_test_from_save_pretrained_dynamo, inputs=None)

    def test_from_pretrained_hub(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        ddpm = DDPMPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm = ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub = ddpm_from_hub.to(torch_device)
        ddpm_from_hub.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm(generator=generator, num_inference_steps=5, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, num_inference_steps=5, output_type="np").images

        assert np.abs(image - new_image).max() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub_pass_model(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDPMScheduler(num_train_timesteps=10)

        # pass unet into DiffusionPipeline
        unet = UNet2DModel.from_pretrained(model_path)
        ddpm_from_hub_custom_model = DiffusionPipeline.from_pretrained(model_path, unet=unet, scheduler=scheduler)
        ddpm_from_hub_custom_model = ddpm_from_hub_custom_model.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path, scheduler=scheduler)
        ddpm_from_hub = ddpm_from_hub.to(torch_device)
        ddpm_from_hub_custom_model.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(0)
        image = ddpm_from_hub_custom_model(generator=generator, num_inference_steps=5, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(0)
        new_image = ddpm_from_hub(generator=generator, num_inference_steps=5, output_type="np").images

        assert np.abs(image - new_image).max() < 1e-5, "Models don't give the same forward pass"

    def test_output_format(self):
        model_path = "google/ddpm-cifar10-32"

        scheduler = DDIMScheduler.from_pretrained(model_path)
        pipe = DDIMPipeline.from_pretrained(model_path, scheduler=scheduler)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        images = pipe(output_type="np").images
        assert images.shape == (1, 32, 32, 3)
        assert isinstance(images, np.ndarray)

        images = pipe(output_type="pil", num_inference_steps=4).images
        assert isinstance(images, list)
        assert len(images) == 1
        assert isinstance(images[0], PIL.Image.Image)

        # use PIL by default
        images = pipe(num_inference_steps=4).images
        assert isinstance(images, list)
        assert isinstance(images[0], PIL.Image.Image)

    @require_flax
    def test_from_flax_from_pt(self):
        pipe_pt = StableDiffusionPipeline.from_pretrained(
            "hf-internal-testing/tiny-stable-diffusion-torch", safety_checker=None
        )
        pipe_pt.to(torch_device)

        from diffusers import FlaxStableDiffusionPipeline

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe_pt.save_pretrained(tmpdirname)

            pipe_flax, params = FlaxStableDiffusionPipeline.from_pretrained(
                tmpdirname, safety_checker=None, from_pt=True
            )

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe_flax.save_pretrained(tmpdirname, params=params)
            pipe_pt_2 = StableDiffusionPipeline.from_pretrained(tmpdirname, safety_checker=None, from_flax=True)
            pipe_pt_2.to(torch_device)

        prompt = "Hello"

        generator = torch.manual_seed(0)
        image_0 = pipe_pt(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images[0]

        generator = torch.manual_seed(0)
        image_1 = pipe_pt_2(
            [prompt],
            generator=generator,
            num_inference_steps=2,
            output_type="np",
        ).images[0]

        assert np.abs(image_0 - image_1).sum() < 1e-5, "Models don't give the same forward pass"

    @require_compel
    def test_weighted_prompts_compel(self):
        from compel import Compel

        pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

        compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

        prompt = "a red cat playing with a ball{}"

        prompts = [prompt.format(s) for s in ["", "++", "--"]]

        prompt_embeds = compel(prompts)

        generator = [torch.Generator(device="cpu").manual_seed(33) for _ in range(prompt_embeds.shape[0])]

        images = pipe(
            prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20, output_type="np"
        ).images

        for i, image in enumerate(images):
            expected_image = load_numpy(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
                f"/compel/forest_{i}.npy"
            )

            assert np.abs(image - expected_image).max() < 3e-1


@nightly
@require_torch_gpu
class PipelineNightlyTests(unittest.TestCase):
    def setUp(self):
        # clean up the VRAM before each test
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_ddpm_ddim_equality_batched(self):
        seed = 0
        model_id = "google/ddpm-cifar10-32"

        unet = UNet2DModel.from_pretrained(model_id)
        ddpm_scheduler = DDPMScheduler()
        ddim_scheduler = DDIMScheduler()

        ddpm = DDPMPipeline(unet=unet, scheduler=ddpm_scheduler)
        ddpm.to(torch_device)
        ddpm.set_progress_bar_config(disable=None)

        ddim = DDIMPipeline(unet=unet, scheduler=ddim_scheduler)
        ddim.to(torch_device)
        ddim.set_progress_bar_config(disable=None)

        generator = torch.Generator(device=torch_device).manual_seed(seed)
        ddpm_images = ddpm(batch_size=2, generator=generator, output_type="np").images

        generator = torch.Generator(device=torch_device).manual_seed(seed)
        ddim_images = ddim(
            batch_size=2,
            generator=generator,
            num_inference_steps=1000,
            eta=1.0,
            output_type="np",
            use_clipped_model_output=True,  # Need this to make DDIM match DDPM
        ).images

        # the values aren't exactly equal, but the images look the same visually
        assert np.abs(ddpm_images - ddim_images).max() < 1e-1