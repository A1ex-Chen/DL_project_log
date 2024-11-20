import argparse
import atexit
import inspect
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import pycuda.driver as cuda
import tensorrt as trt
import torch
from PIL import Image
from pycuda.tools import make_default_context
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


# Initialize CUDA
cuda.init()
context = make_default_context()
device = context.get_device()
atexit.register(context.pop)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name




class TensorRTModel:
    def __init__(
        self,
        trt_engine_path,
        **kwargs,
    ):
        cuda.init()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        trt.init_libnvinfer_plugins(TRT_LOGGER, "")
        trt_runtime = trt.Runtime(TRT_LOGGER)
        engine = load_engine(trt_runtime, trt_engine_path)
        context = engine.create_execution_context()

        # allocates memory for network inputs/outputs on both CPU and GPU
        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        input_names = []
        output_names = []

        for binding in engine:
            datatype = engine.get_binding_dtype(binding)
            if datatype == trt.DataType.HALF:
                dtype = np.float16
            else:
                dtype = np.float32

            shape = tuple(engine.get_binding_shape(binding))
            host_mem = cuda.pagelocked_empty(shape, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))

            if engine.binding_is_input(binding):
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
                input_names.append(binding)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)
                output_names.append(binding)

        self.stream = stream
        self.context = context
        self.engine = engine

        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

        self.input_names = input_names
        self.output_names = output_names

    def __call__(self, **kwargs):
        context = self.context
        stream = self.stream
        bindings = self.bindings

        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs

        for idx, input_name in enumerate(self.input_names):
            _input = kwargs[input_name]
            np.copyto(host_inputs[idx], _input)
            # transfer input data to the GPU
            cuda.memcpy_htod_async(cuda_inputs[idx], host_inputs[idx], stream)

        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        result = {}
        for idx, output_name in enumerate(self.output_names):
            # transfer predictions back from the GPU
            cuda.memcpy_dtoh_async(host_outputs[idx], cuda_outputs[idx], stream)
            result[output_name] = host_outputs[idx]

        stream.synchronize()

        return result


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


def prepare_image(image):
    if isinstance(image, torch.Tensor):
        # Batch single image
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(dtype=torch.float32)
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image


class TensorRTStableDiffusionControlNetImg2ImgPipeline(DiffusionPipeline):
    vae_encoder: OnnxRuntimeModel
    vae_decoder: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: TensorRTModel
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

    parser.add_argument(
        "--unet_engine_path",
        type=str,
        required=True,
        help="Path to the unet + controlnet tensorrt model",
    )

    parser.add_argument("--qr_img_path", type=str, required=True, help="Path to the qr code image")

    args = parser.parse_args()

    qr_image = Image.open(args.qr_img_path)
    qr_image = qr_image.resize((512, 512))

    # init stable diffusion pipeline
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(args.sd_model)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    onnx_pipeline = TensorRTStableDiffusionControlNetImg2ImgPipeline(
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
        unet=TensorRTModel(args.unet_engine_path),
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