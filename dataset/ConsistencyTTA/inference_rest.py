import os
import json
import time
import argparse
import soundfile as sf
from tqdm import tqdm

import torch
import wandb

from diffusers import DDPMScheduler, DDIMScheduler, HeunDiscreteScheduler
from audioldm_eval import EvaluationHelper
from models import AudioGDM, AudioLCM, AudioLCM_FTVAE
from tools.build_pretrained import build_pretrained_models
from tools.torch_tools import seed_all


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__








if __name__ == "__main__":
    main()