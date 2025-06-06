def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f'`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`: {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle maximal {self.config.num_train_timesteps} timesteps.'
            )
    self.num_inference_steps = num_inference_steps
    if self.config.timestep_spacing == 'leading':
        step_ratio = (self.config.num_train_timesteps // self.
            num_inference_steps)
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round(
            ).copy().astype(np.int64)
        timesteps += self.config.steps_offset
    elif self.config.timestep_spacing == 'trailing':
        step_ratio = self.config.num_train_timesteps / self.num_inference_steps
        timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, 
            -step_ratio)[::-1]).astype(np.int64)
        timesteps -= 1
    else:
        raise ValueError(
            f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )
    self.timesteps = torch.from_numpy(timesteps).to(device)
