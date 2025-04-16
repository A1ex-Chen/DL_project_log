# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Benchmark a YOLO model formats for speed and accuracy.

Usage:
    from ultralytics.utils.benchmarks import ProfileModels, benchmark
    ProfileModels(['yolov8n.yaml', 'yolov8s.yaml']).profile()
    benchmark(model='yolov8n.pt', imgsz=160)

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlpackage
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/
NCNN                    | `ncnn`                    | yolov8n_ncnn_model/
"""

import glob
import os
import platform
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch.cuda
import yaml

from ultralytics import YOLO, YOLOWorld
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.utils import ARM64, ASSETS, IS_JETSON, IS_RASPBERRYPI, LINUX, LOGGER, MACOS, TQDM, WEIGHTS_DIR
from ultralytics.utils.checks import IS_PYTHON_3_12, check_requirements, check_yolo
from ultralytics.utils.downloads import safe_download
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device




class RF100Benchmark:



    @staticmethod



class ProfileModels:
    """
    ProfileModels class for profiling different models on ONNX and TensorRT.

    This class profiles the performance of different models, returning results such as model speed and FLOPs.

    Attributes:
        paths (list): Paths of the models to profile.
        num_timed_runs (int): Number of timed runs for the profiling. Default is 100.
        num_warmup_runs (int): Number of warmup runs before profiling. Default is 10.
        min_time (float): Minimum number of seconds to profile for. Default is 60.
        imgsz (int): Image size used in the models. Default is 640.

    Methods:
        profile(): Profiles the models and prints the result.

    Example:
        ```python
        from ultralytics.utils.benchmarks import ProfileModels

        ProfileModels(['yolov8n.yaml', 'yolov8s.yaml'], imgsz=640).profile()
        ```
    """





    @staticmethod




    @staticmethod

    @staticmethod