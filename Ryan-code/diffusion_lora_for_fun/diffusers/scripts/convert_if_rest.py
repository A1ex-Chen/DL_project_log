import argparse
import inspect
import os

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModelWithProjection, T5EncoderModel, T5Tokenizer

from diffusers import DDPMScheduler, IFPipeline, IFSuperResolutionPipeline, UNet2DConditionModel
from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker




























# TODO maybe document and/or can do more efficiently (build indices in for loop and extract once for each split?)




# below is copy and pasted from original convert_if_stage_2.py script














if __name__ == "__main__":
    main(parse_args())