def set_timesteps(self, num_inference_steps: int=None, timesteps: Optional[
    List[int]]=None, device: Union[str, torch.device]=None):
    """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`Dict[float, int]`):
                the number of diffusion steps used when generating samples with a pre-trained model. If passed, then
                `timesteps` must be `None`.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps are moved to. {2 / 3: 20, 0.0: 10}
        """
    if timesteps is None:
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1,
            device=device)
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.Tensor(timesteps).to(device)
    self.timesteps = timesteps
