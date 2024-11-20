#!/usr/bin/env python3

"""
Example to run pyprof with imagenet models.
"""

import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.cuda.profiler as profiler
import argparse

from apex import pyprof
from apex.optimizers import FusedAdam, FP16_Optimizer
import fused_adam_cuda


d = {
	"alexnet":				{'H': 224, 'W': 224, 'opts': {}},

	"densenet121":			{'H': 224, 'W': 224, 'opts': {}},
	"densenet161":			{'H': 224, 'W': 224, 'opts': {}},
	"densenet169":			{'H': 224, 'W': 224, 'opts': {}},
	"densenet201":			{'H': 224, 'W': 224, 'opts': {}},

	"googlenet":			{'H': 224, 'W': 224, 'opts': {'aux_logits': False}},

	"mnasnet0_5":			{'H': 224, 'W': 224, 'opts': {}},
	"mnasnet0_75":			{'H': 224, 'W': 224, 'opts': {}},
	"mnasnet1_0":			{'H': 224, 'W': 224, 'opts': {}},
	"mnasnet1_3":			{'H': 224, 'W': 224, 'opts': {}},

	"mobilenet_v2":			{'H': 224, 'W': 224, 'opts': {}},

	"resnet18":				{'H': 224, 'W': 224, 'opts': {}},
	"resnet34":				{'H': 224, 'W': 224, 'opts': {}},
	"resnet50":				{'H': 224, 'W': 224, 'opts': {}},
	"resnet101":			{'H': 224, 'W': 224, 'opts': {}},
	"resnet152":			{'H': 224, 'W': 224, 'opts': {}},

	"resnext50_32x4d":		{'H': 224, 'W': 224, 'opts': {}},
	"resnext101_32x8d":		{'H': 224, 'W': 224, 'opts': {}},

	"wide_resnet50_2":		{'H': 224, 'W': 224, 'opts': {}},
	"wide_resnet101_2":		{'H': 224, 'W': 224, 'opts': {}},

	"shufflenet_v2_x0_5": 	{'H': 224, 'W': 224, 'opts': {}},
	"shufflenet_v2_x1_0": 	{'H': 224, 'W': 224, 'opts': {}},
	"shufflenet_v2_x1_5": 	{'H': 224, 'W': 224, 'opts': {}},
	"shufflenet_v2_x2_0":	{'H': 224, 'W': 224, 'opts': {}},

	"squeezenet1_0":		{'H': 224, 'W': 224, 'opts': {}},
	"squeezenet1_1":		{'H': 224, 'W': 224, 'opts': {}},

	"vgg11":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg11_bn":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg13":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg13_bn":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg16":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg16_bn":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg19":				{'H': 224, 'W': 224, 'opts': {}},
	"vgg19_bn":				{'H': 224, 'W': 224, 'opts': {}},

	"inception_v3":			{'H': 299, 'W': 299, 'opts': {'aux_logits': False}},
	}


if __name__ == "__main__":
	main()