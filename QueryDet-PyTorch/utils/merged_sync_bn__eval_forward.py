def _eval_forward(self, inputs):
    scale = self.weight * torch.rsqrt(self.running_var + self.eps)
    bias = self.bias - self.running_mean * scale
    scale = scale.view(1, -1, 1, 1)
    bias = bias.view(1, -1, 1, 1)
    return [(x * scale + bias) for x in inputs]
