import torch

from diffusers import LMSDiscreteScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class LMSDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (LMSDiscreteScheduler,)
    num_inference_steps = 10










