def forward(self, x):
    if not self.residual:
        return self.proj(self.se(self.depsep(x if self.expand is None else
            self.expand(x))))
    b = self.proj(self.se(self.depsep(x if self.expand is None else self.
        expand(x))))
    if self.training:
        if self.drop():
            multiplication_factor = 0.0
        else:
            multiplication_factor = 1.0 / self.survival_prob
    else:
        multiplication_factor = 1.0
    if self.quantized:
        x = self.residual_quantizer(x)
    return torch.add(x, alpha=multiplication_factor, other=b)
