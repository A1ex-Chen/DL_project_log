def _fuse_lora(self, lora_scale: float=1.0, safe_fusing: bool=False):
    if self.lora_layer is None:
        return
    dtype, device = self.weight.data.dtype, self.weight.data.device
    w_orig = self.weight.data.float()
    w_up = self.lora_layer.up.weight.data.float()
    w_down = self.lora_layer.down.weight.data.float()
    if self.lora_layer.network_alpha is not None:
        w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank
    fused_weight = w_orig + lora_scale * torch.bmm(w_up[None, :], w_down[
        None, :])[0]
    if safe_fusing and torch.isnan(fused_weight).any().item():
        raise ValueError(
            f'This LoRA weight seems to be broken. Encountered NaN values when trying to fuse LoRA weights for {self}.LoRA weights will not be fused.'
            )
    self.weight.data = fused_weight.to(device=device, dtype=dtype)
    self.lora_layer = None
    self.w_up = w_up.cpu()
    self.w_down = w_down.cpu()
    self._lora_scale = lora_scale
