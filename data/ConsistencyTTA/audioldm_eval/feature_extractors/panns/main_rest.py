import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../utils"))
import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utilities import (
    create_folder,
    get_filename,
    create_logging,
    Mixup,
    StatisticsContainer,
)
from models import (
    Cnn14,
    Cnn14_no_specaug,
    Cnn14_no_dropout,
    Cnn6,
    Cnn10,
    ResNet22,
    ResNet38,
    ResNet54,
    Cnn14_emb512,
    Cnn14_emb128,
    Cnn14_emb32,
    MobileNetV1,
    MobileNetV2,
    LeeNet11,
    LeeNet24,
    DaiNet19,
    Res1dNet31,
    Res1dNet51,
    Wavegram_Cnn14,
    Wavegram_Logmel_Cnn14,
    Wavegram_Logmel128_Cnn14,
    Cnn14_16k,
    Cnn14_8k,
    Cnn14_mel32,
    Cnn14_mel128,
    Cnn14_mixup_time_domain,
    Cnn14_DecisionLevelMax,
    Cnn14_DecisionLevelAtt,
)
from pytorch_utils import move_data_to_device, count_parameters, count_flops, do_mixup
from data_generator import (
    AudioSetDataset,
    TrainSampler,
    BalancedTrainSampler,
    AlternateTrainSampler,
    EvaluateSampler,
    collate_fn,
)
from evaluate import Evaluator
import config
from losses import get_loss_func




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example of parser. ")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--workspace", type=str, required=True)
    parser_train.add_argument(
        "--data_type",
        type=str,
        default="full_train",
        choices=["balanced_train", "full_train"],
    )
    parser_train.add_argument("--sample_rate", type=int, default=32000)
    parser_train.add_argument("--window_size", type=int, default=1024)
    parser_train.add_argument("--hop_size", type=int, default=320)
    parser_train.add_argument("--mel_bins", type=int, default=64)
    parser_train.add_argument("--fmin", type=int, default=50)
    parser_train.add_argument("--fmax", type=int, default=14000)
    parser_train.add_argument("--model_type", type=str, required=True)
    parser_train.add_argument(
        "--loss_type", type=str, default="clip_bce", choices=["clip_bce"]
    )
    parser_train.add_argument(
        "--balanced",
        type=str,
        default="balanced",
        choices=["none", "balanced", "alternate"],
    )
    parser_train.add_argument(
        "--augmentation", type=str, default="mixup", choices=["none", "mixup"]
    )
    parser_train.add_argument("--batch_size", type=int, default=32)
    parser_train.add_argument("--learning_rate", type=float, default=1e-3)
    parser_train.add_argument("--resume_iteration", type=int, default=0)
    parser_train.add_argument("--early_stop", type=int, default=1000000)
    parser_train.add_argument("--cuda", action="store_true", default=False)

    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == "train":
        train(args)

    else:
        raise Exception("Error argument!")