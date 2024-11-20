import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from .ddim import DDIMSampler
from .util import instantiate_from_config


torch.set_grad_enabled(False)







@torch.no_grad()



# sampler = initialize_model(sys.argv[1], sys.argv[2])
@torch.no_grad()