import torch
import torch.nn as nn
from audioldm.clap.open_clip import create_model
from audioldm.clap.training.data import get_audio_features
import torchaudio
from transformers import RobertaTokenizer
import torch.nn.functional as F


class CLAPAudioEmbeddingClassifierFreev2(nn.Module):







