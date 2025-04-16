def denoise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
    x = self.mixing_coeff * x + (1 - self.mixing_coeff) * y
    y = self.mixing_coeff * y + (1 - self.mixing_coeff) * x
    return [x, y]
