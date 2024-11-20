@property
def default_timestep(self):
    kwargs = dict(self.forward_default_kwargs)
    num_inference_steps = kwargs.get('num_inference_steps', self.
        default_num_inference_steps)
    try:
        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_classes[0](**scheduler_config)
        scheduler.set_timesteps(num_inference_steps)
        timestep = scheduler.timesteps[0]
    except NotImplementedError:
        logger.warning(
            f'The scheduler {self.__class__.__name__} does not implement a `get_scheduler_config` method. `default_timestep` will be set to the default value of 1.'
            )
        timestep = 1
    return timestep
