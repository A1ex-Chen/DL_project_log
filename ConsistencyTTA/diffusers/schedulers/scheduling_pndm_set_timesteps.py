def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
    self.num_inference_steps = num_inference_steps
    step_ratio = self.config.num_train_timesteps // self.num_inference_steps
    self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
    self._timesteps += self.config.steps_offset
    if self.config.skip_prk_steps:
        self.prk_timesteps = np.array([])
        self.plms_timesteps = np.concatenate([self._timesteps[:-1], self.
            _timesteps[-2:-1], self._timesteps[-1:]])[::-1].copy()
    else:
        prk_timesteps = np.array(self._timesteps[-self.pndm_order:]).repeat(2
            ) + np.tile(np.array([0, self.config.num_train_timesteps //
            num_inference_steps // 2]), self.pndm_order)
        self.prk_timesteps = prk_timesteps[:-1].repeat(2)[1:-1][::-1].copy()
        self.plms_timesteps = self._timesteps[:-3][::-1].copy()
    timesteps = np.concatenate([self.prk_timesteps, self.plms_timesteps]
        ).astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.ets = []
    self.counter = 0
    self.cur_model_output = 0
