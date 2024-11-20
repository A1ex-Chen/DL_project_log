import tempfile

import torch

from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)

from .test_schedulers import SchedulerCommonTest


class DEISMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DEISMultistepScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)
















