import torch
import librosa
from clap_module import create_model
from training.data import get_audio_features
from training.data import int16_to_float32, float32_to_int16
from transformers import RobertaTokenizer

tokenize = RobertaTokenizer.from_pretrained('roberta-base')


    


if __name__ == "__main__":
    infer_text()
    # infer_audio()