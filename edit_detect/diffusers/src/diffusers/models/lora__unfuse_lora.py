def _unfuse_lora(self):
    if not (getattr(self, 'w_up', None) is not None and getattr(self,
        'w_down', None) is not None):
        return
    fused_weight = self.weight.data
    dtype, device = fused_weight.dtype, fused_weight.device
    w_up = self.w_up.to(device=device).float()
    w_down = self.w_down.to(device).float()
    unfused_weight = fused_weight.float() - self._lora_scale * torch.bmm(w_up
        [None, :], w_down[None, :])[0]
    self.weight.data = unfused_weight.to(device=device, dtype=dtype)
    self.w_up = None
    self.w_down = None
