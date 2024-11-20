import torch

from diffusers import DDIMScheduler

from .test_schedulers import SchedulerCommonTest


class DDIMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDIMScheduler,)
    forward_default_kwargs = (("eta", 0.0), ("num_inference_steps", 50))



















