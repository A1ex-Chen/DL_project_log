def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain. 
        Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples
                with a pre-trained model.
        """
    if num_inference_steps > self.config.num_train_timesteps:
        raise ValueError(
            f'`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`: {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle maximal {self.config.num_train_timesteps} timesteps.'
            )
    self.num_inference_steps = num_inference_steps
    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1
        ].copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.alphas_cumprod = self.alphas_cumprod.to(device)
