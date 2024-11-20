import copy
import numpy as np
import os
from peft import TaskType
import torch
import torch.nn as nn
from torch.nn.utils import rnn
from torchvision import transforms
from typing import List

import model.LAMM.conversations as conversations
from model.LAMM.openlamm import StoppingCriteriaList, LAMMStoppingCriteria, \
    build_one_instance, LAMMPEFTModel, VISION_TAGS
from .moe.layer import MoeLoraLayer, Top2Gating
from .moe import MoeLoraConfig
from .resampler3d import Resampler3D






DET_ANSWER_TEMPLATE = [
    "The {P} position of the image contains an object that can be classified as {C}.",
    "The object present at the {P} coordinate in the image is classified as {C}.",
    "There is an object at the {P} location of the image that can be identified as belonging to the category of {C}.",
    "At the {P} position of the image, there is an object categorized as {C}.",
    "At the {P} of the image, there is an item that falls under the category of {C}.",
    "At the coordinates of {P} position of the image, there exists an object categorized as {C}.",
    "The {P} position of the image features an object that falls under the category of {C}.",
    'There is an object at the {P} position of the image, and its category is {C}.',
    'Upon close inspection of the image, it can be observed that there is an object positioned at {P} that belongs to the {C} category.',
    'At the exact coordinates of {P} in the image, there is an object that can be identified as belonging to the {C} category, and this object stands out from the rest of the objects in the image due to its unique color and pattern.',
    'Scanning through the image, it becomes evident that there is an object at {P} that falls under the {C} category.',
    'By carefully examining the image, one can spot an object at {P} that belongs to the {C} category.',
    'Positioned at {P} within the image is an object that can be classified as belonging to the {C} category, and this object is also the only one in the image that has a specific type of texture and a distinctive shape that sets it apart from the other objects.',
    'Upon careful examination of the image, it can be observed that there is an object positioned precisely at {P} that falls under the {C} category, and this object is also the only one in the image that has a specific type of pattern or design that makes it stand out from the rest of the objects.'
]


class Octavius(LAMMPEFTModel):








    @torch.no_grad()






    # ==============================================
    # inference and evaluation 
    # ==============================================


