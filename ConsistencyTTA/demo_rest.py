import os
import json
from time import time
import argparse
import soundfile as sf
import numpy as np
import torch

from diffusers import DDIMScheduler, HeunDiscreteScheduler
from models import AudioLCM, AudioLCM_FTVAE
from tools.build_pretrained import build_pretrained_models
from tools.torch_tools import seed_all
from inference import dotdict


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)






if __name__ == "__main__":
    main()