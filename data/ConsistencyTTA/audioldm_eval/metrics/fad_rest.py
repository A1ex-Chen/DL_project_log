"""
Calculate Frechet Audio Distance betweeen two audio directories.
Frechet distance implementation adapted from: https://github.com/mseitzer/pytorch-fid
VGGish adapted from: https://github.com/harritaylor/torchvggish
"""
import os
import numpy as np
import torch
from torch import nn
from scipy import linalg
from tqdm import tqdm
import soundfile as sf
import resampy


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)




class FrechetAudioDistance:




