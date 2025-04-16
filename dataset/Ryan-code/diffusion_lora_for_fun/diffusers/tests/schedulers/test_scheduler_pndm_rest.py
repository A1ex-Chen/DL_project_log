import tempfile

import torch

from diffusers import PNDMScheduler

from .test_schedulers import SchedulerCommonTest


class PNDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (PNDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)


















