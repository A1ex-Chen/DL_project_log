import torch
import torch.nn.functional as F
import numpy as np
from .shikra.builder.build_shikra import load_pretrained_shikra
from .shikra import model_args, training_args
from .shikra.dataset.process_function import PlainBoxFormatter
from .shikra.dataset.builder import prepare_interactive, SingleImageInteractive
from .test_base import TestBase

class TestShikra(TestBase):

    

    

