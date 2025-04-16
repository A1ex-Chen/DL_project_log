import numpy as np
import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import logging

# import h5py
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
    "audiostock": ["train", "test"],
    "freesound_no_overlap_noesc50": ["train", "test"],
    "epidemic_sound_effects": ["train", "test"],
    "VGGSound": ["train", "test"],
    "urbansound8k_class_label": ["train", "test"],
    "audioset_t5": ["balanced_train", "unbalanced_train", "eval"],
    "epidemic_sound_effects_t5": ["train", "test"],
    "WavText5K": ["train", "test"],
    "esc50_no_overlap": ["train", "test"],
    "usd8k_no_overlap": ["train", "test"],
    "fsd50k_200_class_label": ["train", "test", "valid"],
}


















# def process_ipc(index_path, classes_num, filename):
#     # load data
#     logging.info("Load Data...............")
#     ipc = [[] for _ in range(classes_num)]
#     with h5py.File(index_path, "r") as f:
#         for i in tqdm(range(len(f["target"]))):
#             t_class = np.where(f["target"][i])[0]
#             for t in t_class:
#                 ipc[t].append(i)
#     print(ipc)
#     np.save(filename, ipc)
#     logging.info("Load Data Succeed...............")














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

