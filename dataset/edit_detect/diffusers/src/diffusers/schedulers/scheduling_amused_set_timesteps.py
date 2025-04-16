def set_timesteps(self, num_inference_steps: int, temperature: Union[int,
    Tuple[int, int], List[int]]=(2, 0), device: Union[str, torch.device]=None):
    self.timesteps = torch.arange(num_inference_steps, device=device).flip(0)
    if isinstance(temperature, (tuple, list)):
        self.temperatures = torch.linspace(temperature[0], temperature[1],
            num_inference_steps, device=device)
    else:
        self.temperatures = torch.linspace(temperature, 0.01,
            num_inference_steps, device=device)
