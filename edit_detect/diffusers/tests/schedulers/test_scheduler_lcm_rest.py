import tempfile
from typing import Dict, List, Tuple

import torch

from diffusers import LCMScheduler
from diffusers.utils.testing_utils import torch_device

from .test_schedulers import SchedulerCommonTest


class LCMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (LCMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 10),)


    @property









    # Override test_add_noise_device because the hardcoded num_inference_steps of 100 doesn't work
    # for LCMScheduler under default settings

    # Override test_from_save_pretrained because it hardcodes a timestep of 1

    # Override test_step_shape because uses 0 and 1 as hardcoded timesteps

    # Override test_set_scheduler_outputs_equivalence since it uses 0 as a hardcoded timestep









        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", 50)

        timestep = self.default_valid_timestep

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            sample = self.dummy_sample
            residual = 0.1 * sample

            scheduler.set_timesteps(num_inference_steps)
            kwargs["generator"] = torch.manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            scheduler.set_timesteps(num_inference_steps)
            kwargs["generator"] = torch.manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    def full_loop(self, num_inference_steps=10, seed=0, **config):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config(**config)
        scheduler = scheduler_class(**scheduler_config)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        generator = torch.manual_seed(seed)

        scheduler.set_timesteps(num_inference_steps)

        for t in scheduler.timesteps:
            residual = model(sample, t)
            sample = scheduler.step(residual, t, sample, generator).prev_sample

        return sample

    def test_full_loop_onestep(self):
        sample = self.full_loop(num_inference_steps=1)

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        # TODO: get expected sum and mean
        assert abs(result_sum.item() - 18.7097) < 1e-3
        assert abs(result_mean.item() - 0.0244) < 1e-3

    def test_full_loop_multistep(self):
        sample = self.full_loop(num_inference_steps=10)

        result_sum = torch.sum(torch.abs(sample))
        result_mean = torch.mean(torch.abs(sample))

        # TODO: get expected sum and mean
        assert abs(result_sum.item() - 197.7616) < 1e-3
        assert abs(result_mean.item() - 0.2575) < 1e-3

    def test_custom_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]

        scheduler.set_timesteps(timesteps=timesteps)

        scheduler_timesteps = scheduler.timesteps

        for i, timestep in enumerate(scheduler_timesteps):
            if i == len(timesteps) - 1:
                expected_prev_t = -1
            else:
                expected_prev_t = timesteps[i + 1]

            prev_t = scheduler.previous_timestep(timestep)
            prev_t = prev_t.item()

            self.assertEqual(prev_t, expected_prev_t)

    def test_custom_timesteps_increasing_order(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 51, 0]

        with self.assertRaises(ValueError, msg="`custom_timesteps` must be in descending order."):
            scheduler.set_timesteps(timesteps=timesteps)

    def test_custom_timesteps_passing_both_num_inference_steps_and_timesteps(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [100, 87, 50, 1, 0]
        num_inference_steps = len(timesteps)

        with self.assertRaises(ValueError, msg="Can only pass one of `num_inference_steps` or `custom_timesteps`."):
            scheduler.set_timesteps(num_inference_steps=num_inference_steps, timesteps=timesteps)

    def test_custom_timesteps_too_large(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        timesteps = [scheduler.config.num_train_timesteps]

        with self.assertRaises(
            ValueError,
            msg="`timesteps` must start before `self.config.train_timesteps`: {scheduler.config.num_train_timesteps}}",
        ):
            scheduler.set_timesteps(timesteps=timesteps)