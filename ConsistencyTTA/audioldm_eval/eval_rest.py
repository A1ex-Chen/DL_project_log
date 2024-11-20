import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
from laion_clap import CLAP_Module

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from audioldm_eval.datasets.load_mel import MelPairedDataset, WaveDataset
from audioldm_eval.metrics.fad import FrechetAudioDistance
from audioldm_eval import calculate_fid, calculate_isc, calculate_kid, calculate_kl
from audioldm_eval.feature_extractors.panns import Cnn14
from audioldm_eval.audio.tools import write_json
import audioldm_eval.audio as Audio

from ssr_eval.metrics import AudioMetrics
from tools.t2a_dataset import T2APairedDataset
from tools.torch_tools import seed_all


device = torch.device(
    "cuda:0" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)




class EvaluationHelper:








