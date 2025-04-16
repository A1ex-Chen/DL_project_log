import os
import sys

import torch

from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLControlNetPipeline,
    T2IAdapter,
    WuerstchenCombinedPipeline,
)
from diffusers.utils import load_image


sys.path.append(".")

from utils import (  # noqa: E402
    BASE_PATH,
    PROMPT,
    BenchmarkInfo,
    benchmark_fn,
    bytes_to_giga_bytes,
    flush,
    generate_csv_dict,
    write_to_csv,
)


RESOLUTION_MAPPING = {
    "runwayml/stable-diffusion-v1-5": (512, 512),
    "lllyasviel/sd-controlnet-canny": (512, 512),
    "diffusers/controlnet-canny-sdxl-1.0": (1024, 1024),
    "TencentARC/t2iadapter_canny_sd14v1": (512, 512),
    "TencentARC/t2i-adapter-canny-sdxl-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-2-1": (768, 768),
    "stabilityai/stable-diffusion-xl-base-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-xl-refiner-1.0": (1024, 1024),
    "stabilityai/sdxl-turbo": (512, 512),
}


class BaseBenchmak:
    pipeline_class = None






class TextToImageBenchmark(BaseBenchmak):
    pipeline_class = AutoPipelineForText2Image





class TurboTextToImageBenchmark(TextToImageBenchmark):



class LCMLoRATextToImageBenchmark(TextToImageBenchmark):
    lora_id = "latent-consistency/lcm-lora-sdxl"






class ImageToImageBenchmark(TextToImageBenchmark):
    pipeline_class = AutoPipelineForImage2Image
    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/1665_Girl_with_a_Pearl_Earring.jpg"
    image = load_image(url).convert("RGB")




class TurboImageToImageBenchmark(ImageToImageBenchmark):



class InpaintingBenchmark(ImageToImageBenchmark):
    pipeline_class = AutoPipelineForInpainting
    mask_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/overture-creations-5sI6fQgYIuo_mask.png"
    mask = load_image(mask_url).convert("RGB")




class IPAdapterTextToImageBenchmark(TextToImageBenchmark):
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png"
    image = load_image(url)




class ControlNetBenchmark(TextToImageBenchmark):
    pipeline_class = StableDiffusionControlNetPipeline
    aux_network_class = ControlNetModel
    root_ckpt = "runwayml/stable-diffusion-v1-5"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_image_condition.png"
    image = load_image(url).convert("RGB")




class ControlNetSDXLBenchmark(ControlNetBenchmark):
    pipeline_class = StableDiffusionXLControlNetPipeline
    root_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"



class T2IAdapterBenchmark(ControlNetBenchmark):
    pipeline_class = StableDiffusionAdapterPipeline
    aux_network_class = T2IAdapter
    root_ckpt = "CompVis/stable-diffusion-v1-4"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_for_adapter.png"
    image = load_image(url).convert("L")



class T2IAdapterSDXLBenchmark(T2IAdapterBenchmark):
    pipeline_class = StableDiffusionXLAdapterPipeline
    root_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_for_adapter_sdxl.png"
    image = load_image(url)
