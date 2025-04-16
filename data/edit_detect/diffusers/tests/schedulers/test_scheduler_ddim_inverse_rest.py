import torch

from diffusers import DDIMInverseScheduler

from .test_schedulers import SchedulerCommonTest


class DDIMInverseSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDIMInverseScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50),)

















