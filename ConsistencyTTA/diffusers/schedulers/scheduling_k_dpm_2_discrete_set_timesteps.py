def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.
    device]=None, num_train_timesteps: Optional[int]=None):
    """
        Sets the timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
    self.num_inference_steps = num_inference_steps
    num_train_timesteps = (num_train_timesteps or self.config.
        num_train_timesteps)
    timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps,
        dtype=float)[::-1].copy()
    sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
    self.log_sigmas = torch.from_numpy(np.log(sigmas)).to(device)
    sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas).to(device=device)
    sigmas_interpol = sigmas.log().lerp(sigmas.roll(1).log(), 0.5).exp()
    self.sigmas = torch.cat([sigmas[:1], sigmas[1:].repeat_interleave(2),
        sigmas[-1:]])
    self.sigmas_interpol = torch.cat([sigmas_interpol[:1], sigmas_interpol[
        1:].repeat_interleave(2), sigmas_interpol[-1:]])
    self.init_noise_sigma = self.sigmas.max()
    if str(device).startswith('mps'):
        timesteps = torch.from_numpy(timesteps).to(device, dtype=torch.float32)
    else:
        timesteps = torch.from_numpy(timesteps).to(device)
    timesteps_interpol = self.sigma_to_t(sigmas_interpol).to(device)
    interleaved_timesteps = torch.stack((timesteps_interpol[1:-1, None],
        timesteps[1:, None]), dim=-1).flatten()
    self.timesteps = torch.cat([timesteps[:1], interleaved_timesteps])
    self.sample = None
