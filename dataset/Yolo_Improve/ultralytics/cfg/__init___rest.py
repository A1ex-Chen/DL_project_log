# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_YAML,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    checks,
    colorstr,
    deprecation_warn,
    yaml_load,
    yaml_print,
)

# Define valid tasks and modes
MODES = {"train", "val", "predict", "export", "track", "benchmark"}
TASKS = {"detect", "segment", "classify", "pose", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolov8n.pt",
    "segment": "yolov8n-seg.pt",
    "classify": "yolov8n-cls.pt",
    "pose": "yolov8n-pose.pt",
    "obb": "yolov8n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
CLI_HELP_MSG = f"""
    Arguments received: {str(['yolo'] + ARGV[1:])}. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {TASKS}
                MODE (required) is one of {MODES}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

    5. Explore your datasets using semantic search and SQL with a simple GUI powered by Ultralytics Explorer API
        yolo explorer
    
    6. Streamlit real-time object detection on your webcam with Ultralytics YOLOv8
        yolo streamlit-predict
        
    7. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Define keys for arg type checks
CFG_FLOAT_KEYS = {  # integer or float arguments, i.e. x=2 and x=2.0
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
}
CFG_FRACTION_KEYS = {  # fractional float arguments with 0.0<=values<=1.0
    "dropout",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}
CFG_INT_KEYS = {  # integer-only arguments
    "epochs",
    "patience",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {  # boolean-only arguments
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}






























# Special modes --------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    # Example: entrypoint(debug='yolo predict model=yolov8n.pt')
    entrypoint(debug="")