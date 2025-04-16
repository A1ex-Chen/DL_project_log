def conv2d_hook(self, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.
        in_channels / self.groups) * (2 if multiply_adds else 1)
    bias_ops = 1 if self.bias is not None else 0
    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_height * output_width
    list_conv2d.append(flops)
