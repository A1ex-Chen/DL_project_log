import inspect
import tempfile
import unittest
from typing import Dict, List, Tuple

import torch

from diffusers import EDMEulerScheduler

from .test_schedulers import SchedulerCommonTest


class EDMEulerSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (EDMEulerScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)






    # Override test_from_save_pretrined to use EDMEulerScheduler-specific logic

    # Override test_from_save_pretrined to use EDMEulerScheduler-specific logic

    # Override test_from_save_pretrained to use EDMEulerScheduler-specific logic

    @unittest.skip(reason="EDMEulerScheduler does not support beta schedules.")


        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", 50)

        timestep = 0

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler.set_timesteps(num_inference_steps)
            timestep = scheduler.timesteps[0]

            sample = self.dummy_sample
            scaled_sample = scheduler.scale_model_input(sample, timestep)
            residual = 0.1 * scaled_sample

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            scheduler.set_timesteps(num_inference_steps)

            scaled_sample = scheduler.scale_model_input(sample, timestep)
            residual = 0.1 * scaled_sample

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    @unittest.skip(reason="EDMEulerScheduler does not support beta schedules.")
    def test_trained_betas(self):
        pass