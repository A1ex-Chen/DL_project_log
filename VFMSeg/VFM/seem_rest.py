import os
import warnings
import PIL
from PIL import Image
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

# import gradio as gr
import torch
import argparse
# import whisper
import numpy as np

# from gradio import processing_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from utils.visualizer import Visualizer


from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.data import MetadataCatalog

from class_mapping import COCO_TO_NuScenes, COCO_TO_A2D2_SKITTI, COCO_TO_VKITTI_SKITTI

import csv





@torch.no_grad()










    
    # res.save('/Labs/Scripts/3DPC/exp_xmuda_journal/out/nuscenes_lidarseg/usa_singapore/uda/xmuda/images/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460-SEEM.jpg')
    # res.save('/Labs/Scripts/3DPC/exp_xmuda_journal/out/nuscenes_lidarseg/usa_singapore/uda/xmuda/images/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460-SEEM-ORG.jpg')

if __name__ == '__main__':
    main()