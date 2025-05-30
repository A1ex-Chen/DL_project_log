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
import unittest

import numpy as np
import torch
from parameterized import parameterized

from diffusers import (
    AsymmetricAutoencoderKL,
    AutoencoderKL,
    AutoencoderKLTemporalDecoder,
    AutoencoderTiny,
    ConsistencyDecoderVAE,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.loading_utils import load_image
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
from diffusers.utils.torch_utils import randn_tensor

from ..test_modeling_common import ModelTesterMixin, UNetTesterMixin


enable_full_determinism()










class AutoencoderKLTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AutoencoderKL
    main_input_name = "sample"
    base_precision = 1e-2

    @property

    @property

    @property




    @require_torch_accelerator_with_training




class AsymmetricAutoencoderKLTests(ModelTesterMixin, UNetTesterMixin, unittest.TestCase):
    model_class = AsymmetricAutoencoderKL
    main_input_name = "sample"
    base_precision = 1e-2

    @property

    @property

    @property





class AutoencoderTinyTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderTiny
    main_input_name = "sample"
    base_precision = 1e-2

    @property

    @property

    @property




class ConsistencyDecoderVAETests(ModelTesterMixin, unittest.TestCase):
    model_class = ConsistencyDecoderVAE
    main_input_name = "sample"
    base_precision = 1e-2
    forward_requires_fresh_args = True


    @property

    @property

    @property


    @unittest.skip

    @unittest.skip


class AutoencoderKLTemporalDecoderFastTests(ModelTesterMixin, unittest.TestCase):
    model_class = AutoencoderKLTemporalDecoder
    main_input_name = "sample"
    base_precision = 1e-2

    @property

    @property

    @property




    @unittest.skipIf(torch_device == "mps", "Gradient checkpointing skipped on MPS")


@slow
class AutoencoderTinyIntegrationTests(unittest.TestCase):




    @parameterized.expand(
        [
            [(1, 4, 73, 97), (1, 3, 584, 776)],
            [(1, 4, 97, 73), (1, 3, 776, 584)],
            [(1, 4, 49, 65), (1, 3, 392, 520)],
            [(1, 4, 65, 49), (1, 3, 520, 392)],
            [(1, 4, 49, 49), (1, 3, 392, 392)],
        ]
    )


    @parameterized.expand([(True,), (False,)])


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):





    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824],
                [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824],
            ],
            [
                47,
                [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089],
                [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131],
            ],
            # fmt: on
        ]
    )

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.0513, 0.0289, 1.3799, 0.2166, -0.2573, -0.0871, 0.5103, -0.0999]],
            [47, [-0.4128, -0.1320, -0.3704, 0.1965, -0.4116, -0.2332, -0.3340, 0.2247]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.1609, 0.9866, -0.0487, -0.0777, -0.2716, 0.8368, -0.2055, -0.0814],
                [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824],
            ],
            [
                47,
                [-0.2377, 0.1147, 0.1333, -0.4841, -0.2506, -0.0805, -0.0491, -0.4085],
                [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131],
            ],
            # fmt: on
        ]
    )

    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.2051, -0.1803, -0.2311, -0.2114, -0.3292, -0.3574, -0.2953, -0.3323]],
            [37, [-0.2632, -0.2625, -0.2199, -0.2741, -0.4539, -0.4990, -0.3720, -0.4925]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps

    @parameterized.expand(
        [
            # fmt: off
            [27, [-0.0369, 0.0207, -0.0776, -0.0682, -0.1747, -0.1930, -0.1465, -0.2039]],
            [16, [-0.1628, -0.2134, -0.2747, -0.2642, -0.3774, -0.4404, -0.3687, -0.4277]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16

    @parameterized.expand([(13,), (16,), (27,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )

    @parameterized.expand([(13,), (16,), (37,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.3001, 0.0918, -2.6984, -3.9720, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.5030, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
            # fmt: on
        ]
    )


@slow
class AsymmetricAutoencoderKLIntegrationTests(unittest.TestCase):





    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 0.3552, -0.1337, 0.1078],
                [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824],
            ],
            [
                47,
                [0.4400, 0.0543, 0.2873, 0.2946, 0.0553, 0.0839, -0.1585, 0.2529],
                [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089],
            ],
            # fmt: on
        ]
    )

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.0340, 0.2870, 0.1698, -0.0105, -0.3448, 0.3529, -0.1321, 0.1097],
                [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 0.3552, -0.1337, 0.1078],
            ],
            [
                47,
                [0.4397, 0.0550, 0.2873, 0.2946, 0.0567, 0.0855, -0.1580, 0.2531],
                [0.4397, 0.0550, 0.2873, 0.2946, 0.0567, 0.0855, -0.1580, 0.2531],
            ],
            # fmt: on
        ]
    )

    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.0521, -0.2939, 0.1540, -0.1855, -0.5936, -0.3138, -0.4579, -0.2275]],
            [37, [-0.1820, -0.4345, -0.0455, -0.2923, -0.8035, -0.5089, -0.4795, -0.3106]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps

    @parameterized.expand([(13,), (16,), (37,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.3001, 0.0918, -2.6984, -3.9720, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.5030, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
            # fmt: on
        ]
    )


@slow
class ConsistencyDecoderVAEIntegrationTests(unittest.TestCase):


    @torch.no_grad()





        assert torch_all_close(downscale(sample), downscale(image), atol=0.125)


@slow
class AutoencoderKLIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_sd_vae_model(self, model_id="CompVis/stable-diffusion-v1-4", fp16=False):
        revision = "fp16" if fp16 else None
        torch_dtype = torch.float16 if fp16 else torch.float32

        model = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch_dtype,
            revision=revision,
        )
        model.to(torch_device)

        return model

    def get_generator(self, seed=0):
        generator_device = "cpu" if not torch_device.startswith("cuda") else "cuda"
        if torch_device != "mps":
            return torch.Generator(device=generator_device).manual_seed(seed)
        return torch.manual_seed(seed)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824],
                [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824],
            ],
            [
                47,
                [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089],
                [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131],
            ],
            # fmt: on
        ]
    )
    def test_stable_diffusion(self, seed, expected_slice, expected_slice_mps):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice_mps if torch_device == "mps" else expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.0513, 0.0289, 1.3799, 0.2166, -0.2573, -0.0871, 0.5103, -0.0999]],
            [47, [-0.4128, -0.1320, -0.3704, 0.1965, -0.4116, -0.2332, -0.3340, 0.2247]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_stable_diffusion_fp16(self, seed, expected_slice):
        model = self.get_sd_vae_model(fp16=True)
        image = self.get_sd_image(seed, fp16=True)
        generator = self.get_generator(seed)

        with torch.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, :2, -2:].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-2)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.1609, 0.9866, -0.0487, -0.0777, -0.2716, 0.8368, -0.2055, -0.0814],
                [-0.2395, 0.0098, 0.0102, -0.0709, -0.2840, -0.0274, -0.0718, -0.1824],
            ],
            [
                47,
                [-0.2377, 0.1147, 0.1333, -0.4841, -0.2506, -0.0805, -0.0491, -0.4085],
                [0.0350, 0.0847, 0.0467, 0.0344, -0.0842, -0.0547, -0.0633, -0.1131],
            ],
            # fmt: on
        ]
    )
    def test_stable_diffusion_mode(self, seed, expected_slice, expected_slice_mps):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)

        with torch.no_grad():
            sample = model(image).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice_mps if torch_device == "mps" else expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.2051, -0.1803, -0.2311, -0.2114, -0.3292, -0.3574, -0.2953, -0.3323]],
            [37, [-0.2632, -0.2625, -0.2199, -0.2741, -0.4539, -0.4990, -0.3720, -0.4925]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps
    def test_stable_diffusion_decode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with torch.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=1e-3)

    @parameterized.expand(
        [
            # fmt: off
            [27, [-0.0369, 0.0207, -0.0776, -0.0682, -0.1747, -0.1930, -0.1465, -0.2039]],
            [16, [-0.1628, -0.2134, -0.2747, -0.2642, -0.3774, -0.4404, -0.3687, -0.4277]],
            # fmt: on
        ]
    )
    @require_torch_accelerator_with_fp16
    def test_stable_diffusion_decode_fp16(self, seed, expected_slice):
        model = self.get_sd_vae_model(fp16=True)
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)

        with torch.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand([(13,), (16,), (27,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )
    def test_stable_diffusion_decode_xformers_vs_2_0_fp16(self, seed):
        model = self.get_sd_vae_model(fp16=True)
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64), fp16=True)

        with torch.no_grad():
            sample = model.decode(encoding).sample

        model.enable_xformers_memory_efficient_attention()
        with torch.no_grad():
            sample_2 = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        assert torch_all_close(sample, sample_2, atol=1e-1)

    @parameterized.expand([(13,), (16,), (37,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )
    def test_stable_diffusion_decode_xformers_vs_2_0(self, seed):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with torch.no_grad():
            sample = model.decode(encoding).sample

        model.enable_xformers_memory_efficient_attention()
        with torch.no_grad():
            sample_2 = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        assert torch_all_close(sample, sample_2, atol=1e-2)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.3001, 0.0918, -2.6984, -3.9720, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.5030, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
            # fmt: on
        ]
    )
    def test_stable_diffusion_encode_sample(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            dist = model.encode(image).latent_dist
            sample = dist.sample(generator=generator)

        assert list(sample.shape) == [image.shape[0], 4] + [i // 8 for i in image.shape[2:]]

        output_slice = sample[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        tolerance = 3e-3 if torch_device != "mps" else 1e-2
        assert torch_all_close(output_slice, expected_output_slice, atol=tolerance)


@slow
class AsymmetricAutoencoderKLIntegrationTests(unittest.TestCase):
    def get_file_format(self, seed, shape):
        return f"gaussian_noise_s={seed}_shape={'_'.join([str(s) for s in shape])}.npy"

    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
        dtype = torch.float16 if fp16 else torch.float32
        image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))).to(torch_device).to(dtype)
        return image

    def get_sd_vae_model(self, model_id="cross-attention/asymmetric-autoencoder-kl-x-1-5", fp16=False):
        revision = "main"
        torch_dtype = torch.float32

        model = AsymmetricAutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            revision=revision,
        )
        model.to(torch_device).eval()

        return model

    def get_generator(self, seed=0):
        generator_device = "cpu" if not torch_device.startswith("cuda") else "cuda"
        if torch_device != "mps":
            return torch.Generator(device=generator_device).manual_seed(seed)
        return torch.manual_seed(seed)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 0.3552, -0.1337, 0.1078],
                [-0.1603, 0.9878, -0.0495, -0.0790, -0.2709, 0.8375, -0.2060, -0.0824],
            ],
            [
                47,
                [0.4400, 0.0543, 0.2873, 0.2946, 0.0553, 0.0839, -0.1585, 0.2529],
                [-0.2376, 0.1168, 0.1332, -0.4840, -0.2508, -0.0791, -0.0493, -0.4089],
            ],
            # fmt: on
        ]
    )
    def test_stable_diffusion(self, seed, expected_slice, expected_slice_mps):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            sample = model(image, generator=generator, sample_posterior=True).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice_mps if torch_device == "mps" else expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=5e-3)

    @parameterized.expand(
        [
            # fmt: off
            [
                33,
                [-0.0340, 0.2870, 0.1698, -0.0105, -0.3448, 0.3529, -0.1321, 0.1097],
                [-0.0344, 0.2912, 0.1687, -0.0137, -0.3462, 0.3552, -0.1337, 0.1078],
            ],
            [
                47,
                [0.4397, 0.0550, 0.2873, 0.2946, 0.0567, 0.0855, -0.1580, 0.2531],
                [0.4397, 0.0550, 0.2873, 0.2946, 0.0567, 0.0855, -0.1580, 0.2531],
            ],
            # fmt: on
        ]
    )
    def test_stable_diffusion_mode(self, seed, expected_slice, expected_slice_mps):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)

        with torch.no_grad():
            sample = model(image).sample

        assert sample.shape == image.shape

        output_slice = sample[-1, -2:, -2:, :2].flatten().float().cpu()
        expected_output_slice = torch.tensor(expected_slice_mps if torch_device == "mps" else expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=3e-3)

    @parameterized.expand(
        [
            # fmt: off
            [13, [-0.0521, -0.2939, 0.1540, -0.1855, -0.5936, -0.3138, -0.4579, -0.2275]],
            [37, [-0.1820, -0.4345, -0.0455, -0.2923, -0.8035, -0.5089, -0.4795, -0.3106]],
            # fmt: on
        ]
    )
    @require_torch_accelerator
    @skip_mps
    def test_stable_diffusion_decode(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with torch.no_grad():
            sample = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        output_slice = sample[-1, -2:, :2, -2:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        assert torch_all_close(output_slice, expected_output_slice, atol=2e-3)

    @parameterized.expand([(13,), (16,), (37,)])
    @require_torch_gpu
    @unittest.skipIf(
        not is_xformers_available(),
        reason="xformers is not required when using PyTorch 2.0.",
    )
    def test_stable_diffusion_decode_xformers_vs_2_0(self, seed):
        model = self.get_sd_vae_model()
        encoding = self.get_sd_image(seed, shape=(3, 4, 64, 64))

        with torch.no_grad():
            sample = model.decode(encoding).sample

        model.enable_xformers_memory_efficient_attention()
        with torch.no_grad():
            sample_2 = model.decode(encoding).sample

        assert list(sample.shape) == [3, 3, 512, 512]

        assert torch_all_close(sample, sample_2, atol=5e-2)

    @parameterized.expand(
        [
            # fmt: off
            [33, [-0.3001, 0.0918, -2.6984, -3.9720, -3.2099, -5.0353, 1.7338, -0.2065, 3.4267]],
            [47, [-1.5030, -4.3871, -6.0355, -9.1157, -1.6661, -2.7853, 2.1607, -5.0823, 2.5633]],
            # fmt: on
        ]
    )
    def test_stable_diffusion_encode_sample(self, seed, expected_slice):
        model = self.get_sd_vae_model()
        image = self.get_sd_image(seed)
        generator = self.get_generator(seed)

        with torch.no_grad():
            dist = model.encode(image).latent_dist
            sample = dist.sample(generator=generator)

        assert list(sample.shape) == [image.shape[0], 4] + [i // 8 for i in image.shape[2:]]

        output_slice = sample[0, -1, -3:, -3:].flatten().cpu()
        expected_output_slice = torch.tensor(expected_slice)

        tolerance = 3e-3 if torch_device != "mps" else 1e-2
        assert torch_all_close(output_slice, expected_output_slice, atol=tolerance)


@slow
class ConsistencyDecoderVAEIntegrationTests(unittest.TestCase):
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

    @torch.no_grad()
    def test_encode_decode(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")  # TODO - update
        vae.to(torch_device)

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        ).resize((256, 256))
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)[
            None, :, :, :
        ].cuda()

        latent = vae.encode(image).latent_dist.mean

        sample = vae.decode(latent, generator=torch.Generator("cpu").manual_seed(0)).sample

        actual_output = sample[0, :2, :2, :2].flatten().cpu()
        expected_output = torch.tensor([-0.0141, -0.0014, 0.0115, 0.0086, 0.1051, 0.1053, 0.1031, 0.1024])

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_sd(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder")  # TODO - update
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", vae=vae, safety_checker=None)
        pipe.to(torch_device)

        out = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        actual_output = out[:2, :2, :2].flatten().cpu()
        expected_output = torch.tensor([0.7686, 0.8228, 0.6489, 0.7455, 0.8661, 0.8797, 0.8241, 0.8759])

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_encode_decode_f16(self):
        vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder", torch_dtype=torch.float16
        )  # TODO - update
        vae.to(torch_device)

        image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            "/img2img/sketch-mountains-input.jpg"
        ).resize((256, 256))
        image = (
            torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)[None, :, :, :]
            .half()
            .cuda()
        )

        latent = vae.encode(image).latent_dist.mean

        sample = vae.decode(latent, generator=torch.Generator("cpu").manual_seed(0)).sample

        actual_output = sample[0, :2, :2, :2].flatten().cpu()
        expected_output = torch.tensor(
            [-0.0111, -0.0125, -0.0017, -0.0007, 0.1257, 0.1465, 0.1450, 0.1471],
            dtype=torch.float16,
        )

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_sd_f16(self):
        vae = ConsistencyDecoderVAE.from_pretrained(
            "openai/consistency-decoder", torch_dtype=torch.float16
        )  # TODO - update
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            vae=vae,
            safety_checker=None,
        )
        pipe.to(torch_device)

        out = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        actual_output = out[:2, :2, :2].flatten().cpu()
        expected_output = torch.tensor(
            [0.0000, 0.0249, 0.0000, 0.0000, 0.1709, 0.2773, 0.0471, 0.1035],
            dtype=torch.float16,
        )

        assert torch_all_close(actual_output, expected_output, atol=5e-3)

    def test_vae_tiling(self):
        vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", vae=vae, safety_checker=None, torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        out_1 = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        # make sure tiled vae decode yields the same result
        pipe.enable_vae_tiling()
        out_2 = pipe(
            "horse",
            num_inference_steps=2,
            output_type="pt",
            generator=torch.Generator("cpu").manual_seed(0),
        ).images[0]

        assert torch_all_close(out_1, out_2, atol=5e-3)

        # test that tiled decode works with various shapes
        shapes = [(1, 4, 73, 97), (1, 4, 97, 73), (1, 4, 49, 65), (1, 4, 65, 49)]
        with torch.no_grad():
            for shape in shapes:
                image = torch.zeros(shape, device=torch_device, dtype=pipe.vae.dtype)
                pipe.vae.decode(image)