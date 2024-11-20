import torch

from diffusers import EulerAncestralDiscreteScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class EulerAncestralDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EulerAncestralDiscreteScheduler,)
    num_inference_steps = 10









