def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Note that this scheduler uses a slightly different step ratio than the other diffusers schedulers. The
        different step ratio is to mimic the original karlo implementation and does not affect the quality or accuracy
        of the results.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
    self.num_inference_steps = num_inference_steps
    step_ratio = (self.config.num_train_timesteps - 1) / (self.
        num_inference_steps - 1)
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1
        ].copy().astype(np.int64)
    self.timesteps = torch.from_numpy(timesteps).to(device)
