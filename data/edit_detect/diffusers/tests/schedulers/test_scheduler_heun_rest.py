import torch

from diffusers import HeunDiscreteScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class HeunDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (HeunDiscreteScheduler,)
    num_inference_steps = 10













