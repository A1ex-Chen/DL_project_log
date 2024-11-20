import torch

from diffusers import UnCLIPScheduler

from .test_schedulers import SchedulerCommonTest


# UnCLIPScheduler is a modified DDPMScheduler with a subset of the configuration.
class UnCLIPSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (UnCLIPScheduler,)












