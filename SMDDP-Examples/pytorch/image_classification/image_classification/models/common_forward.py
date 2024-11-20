def forward(self, x):
    if not self.quantized:
        return super().forward(x) * x
    else:
        x_quant = self.mul_a_quantizer(super().forward(x))
        return x_quant * self.mul_b_quantizer(x)
