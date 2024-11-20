def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
    self.num_inference_steps = num_inference_steps
    clipped_idx = torch.searchsorted(torch.flip(self.lambda_t, [0]), self.
        config.lambda_min_clipped)
    timesteps = np.linspace(0, self.config.num_train_timesteps - 1 -
        clipped_idx, num_inference_steps + 1).round()[::-1][:-1].copy().astype(
        np.int64)
    print(timesteps)
    if self.use_karras_sigmas:
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) **
            0.5)
        log_sigmas = np.log(sigmas)
        sigmas = self._convert_to_karras(in_sigmas=sigmas,
            num_inference_steps=num_inference_steps)
        timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in
            sigmas]).round()
        timesteps = np.flip(timesteps).copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.model_outputs = [None] * self.config.solver_order
    self.sample = None
    if (not self.config.lower_order_final and num_inference_steps % self.
        config.solver_order != 0):
        logger.warn(
            'Changing scheduler {self.config} to have `lower_order_final` set to True to handle uneven amount of inference steps. Please make sure to always use an even number of `num_inference steps when using `lower_order_final=True`.'
            )
        self.register_to_config(lower_order_final=True)
    self.order_list = self.get_order_list(num_inference_steps)
