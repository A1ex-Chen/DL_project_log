import os
import argparse
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras





    return step_function

def get_callbacks(args):
    lr_sched_name = args.lr_sched_name
    batch_size = args.batch_size
    initial_lr = args.learning_rate
    callbacks = None
    if(lr_sched_name == "step_function"):
        callbacks = [keras.callbacks.LearningRateScheduler(step_function_wrapper(batch_size),verbose=1)]
    return callbacks


    