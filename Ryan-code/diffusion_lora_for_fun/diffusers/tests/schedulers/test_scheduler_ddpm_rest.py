import torch

from diffusers import DDPMScheduler

from .test_schedulers import SchedulerCommonTest


class DDPMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDPMScheduler,)

















