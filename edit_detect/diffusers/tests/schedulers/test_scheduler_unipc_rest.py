import tempfile

import torch

from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)

from .test_schedulers import SchedulerCommonTest


class UniPCMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (UniPCMultistepScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)





















class UniPCMultistepScheduler1DTest(UniPCMultistepSchedulerTest):
    @property

    @property

    @property





