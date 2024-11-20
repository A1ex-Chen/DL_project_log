def set_timesteps(self, stength, num_inference_steps: int, lcm_origin_steps:
    int, device: Union[str, torch.device]=None):
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
    c = self.config.num_train_timesteps // lcm_origin_steps
    lcm_origin_timesteps = np.asarray(list(range(1, int(lcm_origin_steps *
        stength) + 1))) * c - 1
    skipping_step = len(lcm_origin_timesteps) // num_inference_steps
    timesteps = lcm_origin_timesteps[::-skipping_step][:num_inference_steps]
    self.timesteps = torch.from_numpy(timesteps.copy()).to(device)
