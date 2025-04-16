import torch

from diffusers import SASolverScheduler
from diffusers.utils.testing_utils import require_torchsde, torch_device

from .test_schedulers import SchedulerCommonTest


@require_torchsde
class SASolverSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (SASolverScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)
    num_inference_steps = 10









