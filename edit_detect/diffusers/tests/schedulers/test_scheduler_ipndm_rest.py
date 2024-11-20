import tempfile

import torch

from diffusers import IPNDMScheduler

from .test_schedulers import SchedulerCommonTest


class IPNDMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (IPNDMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)








