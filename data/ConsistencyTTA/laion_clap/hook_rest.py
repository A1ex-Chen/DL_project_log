"""
Contrastive Language-Audio Pretraining Model from LAION
--------------------------------------------------------
Paper: https://arxiv.org/abs/2211.06687
Authors (equal contributions): Ke Chen, Yusong Wu, Tianyu Zhang, Yuchen Hui
Support: LAION
"""
import os
import torch
import librosa
from clap_module import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16

from transformers import RobertaTokenizer
import wget
from clap_module.factory import load_state_dict


class CLAP_Module(torch.nn.Module):


    



        
    