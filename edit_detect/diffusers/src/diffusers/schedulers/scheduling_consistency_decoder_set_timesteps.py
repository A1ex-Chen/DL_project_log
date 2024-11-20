def set_timesteps(self, num_inference_steps: Optional[int]=None, device:
    Union[str, torch.device]=None):
    if num_inference_steps != 2:
        raise ValueError(
            'Currently more than 2 inference steps are not supported.')
    self.timesteps = torch.tensor([1008, 512], dtype=torch.long, device=device)
    self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
    self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(
        device)
    self.c_skip = self.c_skip.to(device)
    self.c_out = self.c_out.to(device)
    self.c_in = self.c_in.to(device)
