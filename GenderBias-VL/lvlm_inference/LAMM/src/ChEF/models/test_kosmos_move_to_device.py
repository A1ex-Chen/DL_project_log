def move_to_device(self, cfg, device):
    self.dtype = torch.float16
    self.device = device
    self.model.to(device=self.device, dtype=self.dtype)
    self.model.prepare_for_inference_(cfg)
