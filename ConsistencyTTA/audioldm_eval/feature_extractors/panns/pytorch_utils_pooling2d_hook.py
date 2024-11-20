def pooling2d_hook(self, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    kernel_ops = self.kernel_size * self.kernel_size
    bias_ops = 0
    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_height * output_width
    list_pooling2d.append(flops)
