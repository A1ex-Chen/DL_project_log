from __future__ import absolute_import, print_function

import glob
import os
import sys
import threading
from importlib import reload

import numpy as np
import tensorflow.keras.backend as K

# import matplotlib
# if 'MACOSX' in matplotlib.get_backend().upper():
#   matplotlib.use('TKAgg')
# import pylab as py
# py.ion() ## Turn on plot visualization
# import gzip, pickle
# from PIL import Image
# import cv2
# from tqdm import *


file_path = os.path.dirname(os.path.realpath(__file__))

import random

import candle
import helper

additional_definitions = [
    {
        "name": "train_bool",
        "type": candle.str2bool,
        "default": True,
        "help": "Invoke training",
    },
    {
        "name": "eval_bool",
        "type": candle.str2bool,
        "default": False,
        "help": "Use model for inference",
    },
    {"name": "home_dir", "help": "Home Directory", "type": str, "default": "."},
    # {'name': 'config_file','help': 'Config File','type':str,'default':os.path.join(file_path, 'p2b1_default_model.txt')},
    {
        "name": "weight_path",
        "help": "Trained Model Pickle File",
        "type": str,
        "default": None,
    },
    {"name": "base_memo", "help": "Memo", "type": str, "default": None},
    # {'name': 'seed_bool', 'type':candle.str2bool,'default':False,'help': 'Random Seed'},
    {"name": "case", "help": "[Full, Center, CenterZ]", "type": str, "default": "Full"},
    {
        "name": "fig_bool",
        "type": candle.str2bool,
        "default": False,
        "help": "Generate Prediction Figure",
    },
    {
        "name": "set_sel",
        "help": "[3k_Disordered, 3k_Ordered, 3k_Ordered_and_gel, 6k_Disordered, 6k_Ordered, 6k_Ordered_and_gel]",
        "type": str,
        "default": "3k_Disordered",
    },
    {
        "name": "conv_bool",
        "type": candle.str2bool,
        "default": True,
        "help": "Invoke training using 1D Convs for inner AE",
    },
    {
        "name": "full_conv_bool",
        "type": candle.str2bool,
        "default": False,
        "help": "Invoke training using fully convolutional NN for inner AE",
    },
    {
        "name": "type_bool",
        "type": candle.str2bool,
        "default": True,
        "help": "Include molecule type information in desining AE",
    },
    {
        "name": "nbr_type",
        "type": str,
        "default": "relative",
        "help": "Defines the type of neighborhood data to use. [relative, invariant]",
    },
    {"name": "backend", "help": "Keras Backend", "type": str, "default": "tensorflow"},
    {
        "name": "cool",
        "help": "Boolean: cool learning rate",
        "type": candle.str2bool,
        "default": False,
    },
    {"name": "data_set", "help": "Data set for training", "type": str, "default": None},
    {
        "name": "l2_reg",
        "help": "Regularization parameter",
        "type": float,
        "default": None,
    },
    {
        "name": "molecular_nbrs",
        "help": "Data dimension for molecular autoencoder",
        "type": int,
        "default": None,
    },
    {
        "name": "molecular_nonlinearity",
        "help": "Activation for molecular netowrk",
        "type": str,
        "default": None,
    },
    {
        "name": "molecular_num_hidden",
        "nargs": "+",
        "help": "Layer sizes for molecular network",
        "type": int,
        "default": None,
    },
    {"name": "noise_factor", "help": "Noise factor", "type": float, "default": None},
    {
        "name": "num_hidden",
        "nargs": "+",
        "help": "Dense layer specification",
        "type": int,
        "default": None,
    },
    {
        "name": "sampling_density",
        "help": "Sampling density",
        "type": float,
        "default": None,
    },
]

required = [
    "num_hidden",
    "batch_size",
    "learning_rate",
    "epochs",
    "l2_reg",
    "noise_factor",
    "optimizer",
    "loss",
    "activation",
    # note 'cool' is a boolean
    "cool",
    "molecular_num_hidden",
    "molecular_nonlinearity",
    "molecular_nbrs",
    "dropout",
    "l2_reg",
    "sampling_density",
    "save_path",
]


class BenchmarkP2B1(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions




# get activations for hidden layers of the model


def get_list_of_data_files(GP):

    import pilot2_datasets as p2

    reload(p2)
    print("Reading Data...")
    # Identify the data set selected
    data_set = p2.data_sets[GP["set_sel"]][0]
    # Get the MD5 hash for the proper data set
    # data_hash = p2.data_sets[GP['set_sel']][1]
    print("Reading Data Files... %s->%s" % (GP["set_sel"], data_set))
    # Check if the data files are in the data director, otherwise fetch from FTP
    data_file = candle.fetch_file(
        "http://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot2/"
        + data_set
        + ".tar.gz",
        unpack=True,
        subdir="Pilot2",
    )
    data_dir = os.path.join(os.path.dirname(data_file), data_set)
    # Make a list of all of the data files in the data set
    data_files = glob.glob("%s/*.npz" % data_dir)

    fields = p2.gen_data_set_dict()

    return (data_files, fields)


# get activations for hidden layers of the model
def get_activations(model, layer, X_batch):
    get_activations = K.function(
        [model.layers[0].input, K.learning_phase()], [model.layers[layer].output]
    )
    activations = get_activations([X_batch, 0])
    return activations


# ############ Define Data Generators ################
class ImageNoiseDataGenerator(object):
    """Generate minibatches with
    realtime data augmentation.
    """


            # if b==None:
            #    return







class autoencoder_preprocess:




class Candle_Molecular_Train:

