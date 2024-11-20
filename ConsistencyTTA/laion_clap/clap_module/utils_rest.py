import numpy as np
import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import logging
import h5py
from tqdm import tqdm
import random
import json
import os
import pathlib

# TODO: (yusong) this not a good place to store those information and does not scale. Need to be fixed later.
dataset_split = {
    "audiocaps": ["train", "valid", "test"],
    "audioset": ["balanced_train", "unbalanced_train", "eval"],
    "BBCSoundEffects": ["train", "test"],
    "Clotho": ["train", "test", "valid"],
    "free_to_use_sounds": ["train", "test"],
    "paramount_motion": ["train", "test"],
    "sonniss_game_effects": ["train", "test"],
    "wesoundeffects": ["train", "test"],
    "MACS": ["train", "test"],
    "freesound": ["train", "test"],
    "FSD50K": ["train", "test", "valid"],
    "fsd50k_class_label": ["train", "test", "valid"],
    "esc50": ["train", "test"],
    "ESC50_1": ["train", "test"],
    "ESC50_2": ["train", "test"],
    "ESC50_3": ["train", "test"],
    "ESC50_4": ["train", "test"],
    "ESC50_5": ["train", "test"],
    "audiostock": ["train", "test"],
    "freesound_no_overlap_noesc50": ["train", "test"],
    "epidemic_sound_effects": ["train", "test"],
    "VGGSound": ["train", "test"],
    "urbansound8k_class_label": ["train", "test"],
    "audioset_t5": ["balanced_train", "unbalanced_train", "eval"],
    "audioset_t5_debiased": ["balanced_train", "unbalanced_train", "eval"],
    "epidemic_sound_effects_t5": ["train", "test"],
    "epidemic_sound_effects_t5_debiased": ["train", "test"],
    "WavText5K": ["train", "test"],
    "esc50_no_overlap": ["train", "test"],
    "usd8k_no_overlap": ["train", "test"],
    "fsd50k_200_class_label": ["train", "test", "valid"],
    "fma_full": ["train", "test"],
    "Genius": ["train", "test"],
    "Jamendo": ["train", "test"],
    "juno": ["train", "test"],
    "CMU_Arctic": ["train", "test"],
    "ravdess": ["train", "test"],
    "Europarl-st": ["train", "test"],
    "common_voice": ["train", "test"],
    "Jamendo_16bit": ["train", "test"],
    "genius_16bit_128": ["train", "test"],
    "juno_16bit": ["train", "test"],
    "fma_full_16bit_128": ["train", "test"],
    "GTZAN": ["train", "test"],
    }
































from multiprocessing import Process, Manager
from multiprocessing import Process, Value, Array
from ctypes import c_wchar


    # if out is None:
    #     return None
    # else:
    #     key = Array(c_wchar, '\n'.join(list(out.keys())), lock=False)
    #     val = Array('i', out.values(), lock=False)
    #     return (key, val)


from torch import optim

