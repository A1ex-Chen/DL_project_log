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
    timesteps = np.linspace(0, self.num_train_timesteps - 1, 
        num_inference_steps + 1).round()[::-1][:-1].copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps).to(device)
    self.model_outputs = [None] * self.config.solver_order
    self.lower_order_nums = 0
