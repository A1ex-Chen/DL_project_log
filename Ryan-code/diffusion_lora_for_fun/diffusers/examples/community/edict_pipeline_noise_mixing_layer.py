def noise_mixing_layer(self, x: torch.Tensor, y: torch.Tensor):
    y = (y - (1 - self.mixing_coeff) * x) / self.mixing_coeff
    x = (x - (1 - self.mixing_coeff) * y) / self.mixing_coeff
    return [x, y]
