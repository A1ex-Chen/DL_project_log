'''
MLCommons
group: TinyMLPerf (https://github.com/mlcommons/tiny)

image classification on cifar10

eval_functions_eembc.py: performances evaluation functions from eembc

refs:
https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Methodology/eval_functions_eembc.py
'''

import numpy as np
import matplotlib.pyplot as plt


# Classifier overall accuracy calculation
# y_pred contains the outputs of the network for the validation data
# labels are the correct answers


# Classifier accuracy per class calculation
# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
# classes are the model's classes


# Classifier ROC AUC calculation
# y_pred contains the outputs of the network for the validation data
# labels are the correct answers
# classes are the model's classes
# name is the model's name


# Classifier overall accuracy calculation
# y_pred contains the outputs of the network for the validation data
# y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
# using this function is not recommended


# Classifier overall accuracy calculation
# y_pred contains the outputs of the network for the validation data
# y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
# this is the function that should be used for accuracy calculations


# Autoencoder ROC AUC calculation
# y_pred contains the outputs of the network for the validation data
# y_true are the correct answers (0.0 for normal, 1.0 for anomaly)
# this is the function that should be used for accuracy calculations
# name is the model's name