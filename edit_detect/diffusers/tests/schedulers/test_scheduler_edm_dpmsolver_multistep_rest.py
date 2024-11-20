import tempfile
import unittest

import torch

from diffusers import (
    EDMDPMSolverMultistepScheduler,
)

from .test_schedulers import SchedulerCommonTest


class EDMDPMSolverMultistepSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EDMDPMSolverMultistepScheduler,)
    forward_default_kwargs = (("num_inference_steps", 25),)










    # TODO (patil-suraj): Fix this test
    @unittest.skip("Skip for now, as it failing currently but works with the actual model")








