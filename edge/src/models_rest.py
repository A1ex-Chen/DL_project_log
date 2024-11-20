import numpy as np
import torch
import torch.nn as nn
import cv2
from utils.general import yaml_load, check_version, LOGGER
from collections import OrderedDict, namedtuple
from utils.torch_utils import smart_inference_mode
import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
import pycuda.driver as cuda
import pycuda.autoinit


class DetectBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    
    @smart_inference_mode()




class BodyFeatureExtractBackend():

            

    