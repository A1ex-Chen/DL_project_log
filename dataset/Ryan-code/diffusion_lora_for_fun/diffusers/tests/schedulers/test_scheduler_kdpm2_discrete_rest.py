import torch

from diffusers import KDPM2DiscreteScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class KDPM2DiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KDPM2DiscreteScheduler,)
    num_inference_steps = 10








