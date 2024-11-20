import tempfile

import torch

from diffusers import DPMSolverMultistepInverseScheduler, DPMSolverMultistepScheduler

from .test_schedulers import SchedulerCommonTest


class DPMSolverMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DPMSolverMultistepInverseScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)





















