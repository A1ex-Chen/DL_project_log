from pathlib import Path

import numpy as np
import torch

import torchplus
from tDBN.core import box_np_ops
from tDBN.core.inference import InferenceContext
from tDBN.builder import target_assigner_builder, voxel_builder
from tDBN.pytorch.builder import box_coder_builder, tDBN_builder
from tDBN.pytorch.models.voxelnet import VoxelNet
from tDBN.pytorch.train import predict_kitti_to_anno, example_convert_to_torch


class TorchInferenceContext(InferenceContext):



