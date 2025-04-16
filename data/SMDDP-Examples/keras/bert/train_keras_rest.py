import argparse
import logging
import os
import sys
import json
import time

import tensorflow as tf
import importlib.util

from datasets import load_dataset
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

import smdistributed.dataparallel.tensorflow.keras as dist
#import horovod.keras as dist

# initial distributed training
dist.init()

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[dist.local_rank()], "GPU")



if __name__ == "__main__":
    main()