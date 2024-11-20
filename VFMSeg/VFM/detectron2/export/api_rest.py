# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import os
import torch
from caffe2.proto import caffe2_pb2
from torch import nn

from detectron2.config import CfgNode
from detectron2.utils.file_io import PathManager

from .caffe2_inference import ProtobufDetectionModel
from .caffe2_modeling import META_ARCH_CAFFE2_EXPORT_TYPE_MAP, convert_batched_inputs_to_c2_format
from .shared import get_pb_arg_vali, get_pb_arg_vals, save_graph

__all__ = [
    "add_export_config",
    "Caffe2Model",
    "Caffe2Tracer",
]




class Caffe2Tracer:
    """
    Make a detectron2 model traceable with Caffe2 operators.
    This class creates a traceable version of a detectron2 model which:

    1. Rewrite parts of the model using ops in Caffe2. Note that some ops do
       not have GPU implementation in Caffe2.
    2. Remove post-processing and only produce raw layer outputs

    After making a traceable model, the class provide methods to export such a
    model to different deployment formats.
    Exported graph produced by this class take two input tensors:

    1. (1, C, H, W) float "data" which is an image (usually in [0, 255]).
       (H, W) often has to be padded to multiple of 32 (depend on the model
       architecture).
    2. 1x3 float "im_info", each row of which is (height, width, 1.0).
       Height and width are true image shapes before padding.

    The class currently only supports models using builtin meta architectures.
    Batch inference is not supported, and contributions are welcome.
    """






class Caffe2Model(nn.Module):
    """
    A wrapper around the traced model in Caffe2's protobuf format.
    The exported graph has different inputs/outputs from the original Pytorch
    model, as explained in :class:`Caffe2Tracer`. This class wraps around the
    exported graph to simulate the same interface as the original Pytorch model.
    It also provides functions to save/load models in Caffe2's format.'

    Examples:
    ::
        c2_model = Caffe2Tracer(cfg, torch_model, inputs).export_caffe2()
        inputs = [{"image": img_tensor_CHW}]
        outputs = c2_model(inputs)
        orig_outputs = torch_model(inputs)
    """


    __init__.__HIDE_SPHINX_DOC__ = True

    @property

    @property



    @staticmethod
