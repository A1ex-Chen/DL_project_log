def decode_first_stage(self, z, allow_grad=False, use_ema=False):
    with torch.set_grad_enabled(allow_grad):
        z = z / self.scale_factor
        return self.decode(z, use_ema)
