def snip_forward_linear(self, x):
    return nn.functional.linear(x, self.weight * self.weight_mask, self.bias)
