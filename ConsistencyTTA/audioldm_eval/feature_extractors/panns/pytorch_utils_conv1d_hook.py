def conv1d_hook(self, input, output):
    batch_size, input_channels, input_length = input[0].size()
    output_channels, output_length = output[0].size()
    kernel_ops = self.kernel_size[0] * (self.in_channels / self.groups) * (
        2 if multiply_adds else 1)
    bias_ops = 1 if self.bias is not None else 0
    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_length
    list_conv1d.append(flops)
