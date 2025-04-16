def forward(self, x):
    print(self.quant_weight)
    w = self.quant_weight
    w_scale = self.weight_scale
    x_norm = self.norm(x)
    x_quant, x_scale = activation_quant(x_norm)
    y = gemm_lowbit_kernel(x_quant, w) / w_scale / x_scale
    return y
