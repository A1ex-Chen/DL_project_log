import torch

from diffusers import TCDScheduler

from .test_schedulers import SchedulerCommonTest


class TCDSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (TCDScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)


    @property

    @property














