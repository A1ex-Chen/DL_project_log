import torch

from diffusers import CMStochasticIterativeScheduler

from .test_schedulers import SchedulerCommonTest


class CMStochasticIterativeSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (CMStochasticIterativeScheduler,)
    num_inference_steps = 10


    # Override test_step_shape to add CMStochasticIterativeScheduler-specific logic regarding timesteps
    # Problem is that we don't know two timesteps that will always be in the timestep schedule from only the scheduler
    # config; scaled sigma_max is always in the timestep schedule, but sigma_min is in the sigma schedule while scaled
    # sigma_min is not in the timestep schedule







