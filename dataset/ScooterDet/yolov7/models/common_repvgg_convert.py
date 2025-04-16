def repvgg_convert(self):
    kernel, bias = self.get_equivalent_kernel_bias()
    return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()
