import sys

import os
import torch
import librosa
from open_clip import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer

tokenize = RobertaTokenizer.from_pretrained("roberta-base")




PRETRAINED_PATH = "/mnt/fast/nobackup/users/hl01486/projects/contrastive_pretraining/CLAP/assets/checkpoints/epoch_top_0_audioset_no_fusion.pt"
WAVE_48k_PATH = "/mnt/fast/nobackup/users/hl01486/projects/contrastive_pretraining/CLAP/assets/audio/machine.wav"






if __name__ == "__main__":
    infer_text()
    infer_audio()