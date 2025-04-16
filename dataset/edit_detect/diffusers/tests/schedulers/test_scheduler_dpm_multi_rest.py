import tempfile

import torch

from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)

from .test_schedulers import SchedulerCommonTest


class DPMSolverMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DPMSolverMultistepScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)


























