import torch

from diffusers import KDPM2AncestralDiscreteScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class KDPM2AncestralDiscreteSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (KDPM2AncestralDiscreteScheduler,)
    num_inference_steps = 10








