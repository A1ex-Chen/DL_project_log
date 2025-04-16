# coding=utf-8
# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import json
import os
import tempfile
import unittest
import uuid
from typing import Dict, List, Tuple

import numpy as np
import torch
from huggingface_hub import delete_repo

import diffusers
from diffusers import (
    CMStochasticIterativeScheduler,
    DDIMScheduler,
    DEISMultistepScheduler,
    DiffusionPipeline,
    EDMEulerScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    IPNDMScheduler,
    LMSDiscreteScheduler,
    UniPCMultistepScheduler,
    VQDiffusionScheduler,
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import logging
from diffusers.utils.testing_utils import CaptureLogger, torch_device

from ..others.test_utils import TOKEN, USER, is_staging_test


torch.backends.cuda.matmul.allow_tf32 = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SchedulerObject(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config


class SchedulerObject2(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config


class SchedulerObject3(SchedulerMixin, ConfigMixin):
    config_name = "config.json"

    @register_to_config


class SchedulerBaseTests(unittest.TestCase):






class SchedulerCommonTest(unittest.TestCase):
    scheduler_classes = ()
    forward_default_kwargs = ()

    @property

    @property

    # NOTE: currently taking the convention that default_timestep > default_timestep_2 (alternatively,
    # default_timestep comes earlier in the timestep schedule than default_timestep_2)
    @property

    @property

    @property

    @property
















@is_staging_test
class SchedulerPushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-scheduler-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"



        return model

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)
        time_step = time_step if time_step is not None else self.default_timestep

        for scheduler_class in self.scheduler_classes:
            # TODO(Suraj) - delete the following two lines once DDPM, DDIM, and PNDM have timesteps casted to float by default
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                time_step = float(time_step)

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == CMStochasticIterativeScheduler:
                # Get valid timestep based on sigma_max, which should always be in timestep schedule.
                scaled_sigma_max = scheduler.sigma_to_t(scheduler.config.sigma_max)
                time_step = scaled_sigma_max

            if scheduler_class == EDMEulerScheduler:
                time_step = scheduler.timesteps[-1]

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, time_step)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Make sure `scale_model_input` is invoked to prevent a warning
            if scheduler_class == CMStochasticIterativeScheduler:
                # Get valid timestep based on sigma_max, which should always be in timestep schedule.
                _ = scheduler.scale_model_input(sample, scaled_sigma_max)
                _ = new_scheduler.scale_model_input(sample, scaled_sigma_max)
            elif scheduler_class != VQDiffusionScheduler:
                _ = scheduler.scale_model_input(sample, scheduler.timesteps[-1])
                _ = new_scheduler.scale_model_input(sample, scheduler.timesteps[-1])

            # Set the seed before step() as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)
        time_step = time_step if time_step is not None else self.default_timestep

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                time_step = float(time_step)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, time_step)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(residual, time_step, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", self.default_num_inference_steps)

        for scheduler_class in self.scheduler_classes:
            timestep = self.default_timestep
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep = float(timestep)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == CMStochasticIterativeScheduler:
                # Get valid timestep based on sigma_max, which should always be in timestep schedule.
                timestep = scheduler.sigma_to_t(scheduler.config.sigma_max)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            output = scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            new_output = new_scheduler.step(residual, timestep, sample, **kwargs).prev_sample

            assert torch.sum(torch.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_compatibles(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()

            scheduler = scheduler_class(**scheduler_config)

            assert all(c is not None for c in scheduler.compatibles)

            for comp_scheduler_cls in scheduler.compatibles:
                comp_scheduler = comp_scheduler_cls.from_config(scheduler.config)
                assert comp_scheduler is not None

            new_scheduler = scheduler_class.from_config(comp_scheduler.config)

            new_scheduler_config = {k: v for k, v in new_scheduler.config.items() if k in scheduler.config}
            scheduler_diff = {k: v for k, v in new_scheduler.config.items() if k not in scheduler.config}

            # make sure that configs are essentially identical
            assert new_scheduler_config == dict(scheduler.config)

            # make sure that only differences are for configs that are not in init
            init_keys = inspect.signature(scheduler_class.__init__).parameters.keys()
            assert set(scheduler_diff.keys()).intersection(set(init_keys)) == set()

    def test_from_pretrained(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()

            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_pretrained(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            # `_use_default_values` should not exist for just saved & loaded scheduler
            scheduler_config = dict(scheduler.config)
            del scheduler_config["_use_default_values"]

            assert scheduler_config == new_scheduler.config

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", self.default_num_inference_steps)

        timestep_0 = self.default_timestep
        timestep_1 = self.default_timestep_2

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep_0 = float(timestep_0)
                timestep_1 = float(timestep_1)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep_0)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(residual, timestep_0, sample, **kwargs).prev_sample
            output_1 = scheduler.step(residual, timestep_1, sample, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_scheduler_outputs_equivalence(self):


        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", self.default_num_inference_steps)

        timestep = self.default_timestep
        if len(self.scheduler_classes) > 0 and self.scheduler_classes[0] == IPNDMScheduler:
            timestep = 1

        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, LMSDiscreteScheduler):
                timestep = float(timestep)

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class == CMStochasticIterativeScheduler:
                # Get valid timestep based on sigma_max, which should always be in timestep schedule.
                timestep = scheduler.sigma_to_t(scheduler.config.sigma_max)

            if scheduler_class == VQDiffusionScheduler:
                num_vec_classes = scheduler_config["num_vec_classes"]
                sample = self.dummy_sample(num_vec_classes)
                model = self.dummy_model(num_vec_classes)
                residual = model(sample, timestep)
            else:
                sample = self.dummy_sample
                residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_dict = scheduler.step(residual, timestep, sample, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            # Set the seed before state as some schedulers are stochastic like EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
            if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
                kwargs["generator"] = torch.manual_seed(0)
            outputs_tuple = scheduler.step(residual, timestep, sample, return_dict=False, **kwargs)

            recursive_check(outputs_tuple, outputs_dict)

    def test_scheduler_public_api(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            if scheduler_class != VQDiffusionScheduler:
                self.assertTrue(
                    hasattr(scheduler, "init_noise_sigma"),
                    f"{scheduler_class} does not implement a required attribute `init_noise_sigma`",
                )
                self.assertTrue(
                    hasattr(scheduler, "scale_model_input"),
                    (
                        f"{scheduler_class} does not implement a required class method `scale_model_input(sample,"
                        " timestep)`"
                    ),
                )
            self.assertTrue(
                hasattr(scheduler, "step"),
                f"{scheduler_class} does not implement a required class method `step(...)`",
            )

            if scheduler_class != VQDiffusionScheduler:
                sample = self.dummy_sample
                if scheduler_class == CMStochasticIterativeScheduler:
                    # Get valid timestep based on sigma_max, which should always be in timestep schedule.
                    scaled_sigma_max = scheduler.sigma_to_t(scheduler.config.sigma_max)
                    scaled_sample = scheduler.scale_model_input(sample, scaled_sigma_max)
                elif scheduler_class == EDMEulerScheduler:
                    scaled_sample = scheduler.scale_model_input(sample, scheduler.timesteps[-1])
                else:
                    scaled_sample = scheduler.scale_model_input(sample, 0.0)
                self.assertEqual(sample.shape, scaled_sample.shape)

    def test_add_noise_device(self):
        for scheduler_class in self.scheduler_classes:
            if scheduler_class == IPNDMScheduler:
                continue
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            scheduler.set_timesteps(self.default_num_inference_steps)

            sample = self.dummy_sample.to(torch_device)
            if scheduler_class == CMStochasticIterativeScheduler:
                # Get valid timestep based on sigma_max, which should always be in timestep schedule.
                scaled_sigma_max = scheduler.sigma_to_t(scheduler.config.sigma_max)
                scaled_sample = scheduler.scale_model_input(sample, scaled_sigma_max)
            elif scheduler_class == EDMEulerScheduler:
                scaled_sample = scheduler.scale_model_input(sample, scheduler.timesteps[-1])
            else:
                scaled_sample = scheduler.scale_model_input(sample, 0.0)
            self.assertEqual(sample.shape, scaled_sample.shape)

            noise = torch.randn_like(scaled_sample).to(torch_device)
            t = scheduler.timesteps[5][None]
            noised = scheduler.add_noise(scaled_sample, noise, t)
            self.assertEqual(noised.shape, scaled_sample.shape)

    def test_deprecated_kwargs(self):
        for scheduler_class in self.scheduler_classes:
            has_kwarg_in_model_class = "kwargs" in inspect.signature(scheduler_class.__init__).parameters
            has_deprecated_kwarg = len(scheduler_class._deprecated_kwargs) > 0

            if has_kwarg_in_model_class and not has_deprecated_kwarg:
                raise ValueError(
                    f"{scheduler_class} has `**kwargs` in its __init__ method but has not defined any deprecated"
                    " kwargs under the `_deprecated_kwargs` class attribute. Make sure to either remove `**kwargs` if"
                    " there are no deprecated arguments or add the deprecated argument with `_deprecated_kwargs ="
                    " [<deprecated_argument>]`"
                )

            if not has_kwarg_in_model_class and has_deprecated_kwarg:
                raise ValueError(
                    f"{scheduler_class} doesn't have `**kwargs` in its __init__ method but has defined deprecated"
                    " kwargs under the `_deprecated_kwargs` class attribute. Make sure to either add the `**kwargs`"
                    f" argument to {self.model_class}.__init__ if there are deprecated arguments or remove the"
                    " deprecated argument from `_deprecated_kwargs = [<deprecated_argument>]`"
                )

    def test_trained_betas(self):
        for scheduler_class in self.scheduler_classes:
            if scheduler_class in (VQDiffusionScheduler, CMStochasticIterativeScheduler):
                continue

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config, trained_betas=np.array([0.1, 0.3]))

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_pretrained(tmpdirname)
                new_scheduler = scheduler_class.from_pretrained(tmpdirname)

            assert scheduler.betas.tolist() == new_scheduler.betas.tolist()

    def test_getattr_is_correct(self):
        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            # save some things to test
            scheduler.dummy_attribute = 5
            scheduler.register_to_config(test_attribute=5)

            logger = logging.get_logger("diffusers.configuration_utils")
            # 30 for warning
            logger.setLevel(30)
            with CaptureLogger(logger) as cap_logger:
                assert hasattr(scheduler, "dummy_attribute")
                assert getattr(scheduler, "dummy_attribute") == 5
                assert scheduler.dummy_attribute == 5

            # no warning should be thrown
            assert cap_logger.out == ""

            logger = logging.get_logger("diffusers.schedulers.scheduling_utils")
            # 30 for warning
            logger.setLevel(30)
            with CaptureLogger(logger) as cap_logger:
                assert hasattr(scheduler, "save_pretrained")
                fn = scheduler.save_pretrained
                fn_1 = getattr(scheduler, "save_pretrained")

                assert fn == fn_1
            # no warning should be thrown
            assert cap_logger.out == ""

            # warning should be thrown
            with self.assertWarns(FutureWarning):
                assert scheduler.test_attribute == 5

            with self.assertWarns(FutureWarning):
                assert getattr(scheduler, "test_attribute") == 5

            with self.assertRaises(AttributeError) as error:
                scheduler.does_not_exist

            assert str(error.exception) == f"'{type(scheduler).__name__}' object has no attribute 'does_not_exist'"


@is_staging_test
class SchedulerPushToHubTester(unittest.TestCase):
    identifier = uuid.uuid4()
    repo_id = f"test-scheduler-{identifier}"
    org_repo_id = f"valid_org/{repo_id}-org"

    def test_push_to_hub(self):
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        scheduler.push_to_hub(self.repo_id, token=TOKEN)
        scheduler_loaded = DDIMScheduler.from_pretrained(f"{USER}/{self.repo_id}")

        assert type(scheduler) == type(scheduler_loaded)

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

        # Push to hub via save_config
        with tempfile.TemporaryDirectory() as tmp_dir:
            scheduler.save_config(tmp_dir, repo_id=self.repo_id, push_to_hub=True, token=TOKEN)

        scheduler_loaded = DDIMScheduler.from_pretrained(f"{USER}/{self.repo_id}")

        assert type(scheduler) == type(scheduler_loaded)

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.repo_id)

    def test_push_to_hub_in_organization(self):
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        scheduler.push_to_hub(self.org_repo_id, token=TOKEN)
        scheduler_loaded = DDIMScheduler.from_pretrained(self.org_repo_id)

        assert type(scheduler) == type(scheduler_loaded)

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)

        # Push to hub via save_config
        with tempfile.TemporaryDirectory() as tmp_dir:
            scheduler.save_config(tmp_dir, repo_id=self.org_repo_id, push_to_hub=True, token=TOKEN)

        scheduler_loaded = DDIMScheduler.from_pretrained(self.org_repo_id)

        assert type(scheduler) == type(scheduler_loaded)

        # Reset repo
        delete_repo(token=TOKEN, repo_id=self.org_repo_id)