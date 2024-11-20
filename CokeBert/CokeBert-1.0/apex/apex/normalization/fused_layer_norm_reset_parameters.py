def reset_parameters(self):
    if self.elementwise_affine:
        init.ones_(self.weight)
        init.zeros_(self.bias)
