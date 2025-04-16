import inspect
import os
from argparse import ArgumentParser

import numpy as np
import torch
from muse import MaskGiTUViT, VQGANModel
from muse import PipelineMuse as OldPipelineMuse
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import VQModel
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.unets.uvit_2d import UVit2DModel
from diffusers.pipelines.amused.pipeline_amused import AmusedPipeline
from diffusers.schedulers import AmusedScheduler


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

# Enable CUDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

device = "cuda"









    # fmt: on


if __name__ == "__main__":
    main()