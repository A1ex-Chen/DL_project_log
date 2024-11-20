import torch

from diffusers import DPMSolverSDEScheduler
from diffusers.utils.testing_utils import require_torchsde, torch_device

from .test_schedulers import SchedulerCommonTest


@require_torchsde
class DPMSolverSDESchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DPMSolverSDEScheduler,)
    num_inference_steps = 10








