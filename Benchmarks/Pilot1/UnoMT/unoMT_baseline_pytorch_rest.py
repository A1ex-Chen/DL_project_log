"""
    File Name:          unoMT_baseline_pytorch.py
    File Description:   This has been taken from the unoMT original
                        scripts (written by Xiaotian Duan, xduan7@uchicago.edu)
                        and has been modified to fit CANDLE framework.
                        Date: 3/12/19.
"""

import datetime

import candle
import numpy as np
import torch
import unoMT
from unoMT_pytorch_model import UnoMTModel
from utils.miscellaneous.random_seeding import seed_random_state

np.set_printoptions(precision=4)








if __name__ == "__main__":
    main()