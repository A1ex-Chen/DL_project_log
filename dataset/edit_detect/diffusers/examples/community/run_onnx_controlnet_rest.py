import argparse
import inspect
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from PIL import Image
from transformers import CLIPTokenizer

from diffusers import OnnxRuntimeModel, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> np_image = np.array(image)

        >>> # get canny image
        >>> np_image = cv2.Canny(np_image, 100, 200)
        >>> np_image = np_image[:, :, None]
        >>> np_image = np.concatenate([np_image, np_image, np_image], axis=2)
        >>> canny_image = Image.fromarray(np_image)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman",
        ...     num_inference_steps=20,
        ...     generator=generator,
        ...     image=image,
        ...     control_image=canny_image,
        ... ).images[0]
        ```
"""




class OnnxStableDiffusionControlNetImg2ImgPipeline(DiffusionPipeline):
    vae_encoder: OnnxRuntimeModel
    vae_decoder: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: OnnxRuntimeModel
    scheduler: KarrasDiffusionSchedulers



    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs


    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.check_image

    # Copied from diffusers.pipelines.controlnet.pipeline_controlnet.StableDiffusionControlNetPipeline.prepare_image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd_model",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--onnx_model_dir",
        type=str,
        required=True,
        help="Path to the ONNX directory",
    )

    parser.add_argument("--qr_img_path", type=str, required=True, help="Path to the qr code image")

    args = parser.parse_args()

    qr_image = Image.open(args.qr_img_path)
    qr_image = qr_image.resize((512, 512))

    # init stable diffusion pipeline
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.sd_model)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    onnx_pipeline = OnnxStableDiffusionControlNetImg2ImgPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(
            os.path.join(args.onnx_model_dir, "vae_encoder"), provider=provider
        ),
        vae_decoder=OnnxRuntimeModel.from_pretrained(
            os.path.join(args.onnx_model_dir, "vae_decoder"), provider=provider
        ),
        text_encoder=OnnxRuntimeModel.from_pretrained(
            os.path.join(args.onnx_model_dir, "text_encoder"), provider=provider
        ),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(os.path.join(args.onnx_model_dir, "unet"), provider=provider),
        scheduler=pipeline.scheduler,
    )
    onnx_pipeline = onnx_pipeline.to("cuda")

    prompt = "a cute cat fly to the moon"
    negative_prompt = "paintings, sketches, worst quality, low quality, normal quality, lowres, normal quality, monochrome, grayscale, skin spots, acnes, skin blemishes, age spot, glans, nsfw, nipples, necklace, worst quality, low quality, watermark, username, signature, multiple breasts, lowres, bad anatomy, bad hands, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, single color, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, disfigured, bad anatomy, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, mutated hands, fused fingers, too many fingers, long neck, bad body perspect"

    for i in range(10):
        start_time = time.time()
        image = onnx_pipeline(
            num_controlnet=2,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=qr_image,
            control_image=[qr_image, qr_image],
            width=512,
            height=512,
            strength=0.75,
            num_inference_steps=20,
            num_images_per_prompt=1,
            controlnet_conditioning_scale=[0.8, 0.8],
            control_guidance_start=[0.3, 0.3],
            control_guidance_end=[0.9, 0.9],
        ).images[0]
        print(time.time() - start_time)
        image.save("output_qr_code.png")