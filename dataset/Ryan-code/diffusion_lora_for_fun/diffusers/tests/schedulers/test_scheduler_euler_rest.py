import torch

from diffusers import EulerDiscreteScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class EulerDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EulerDiscreteScheduler,)
    num_inference_steps = 10

















