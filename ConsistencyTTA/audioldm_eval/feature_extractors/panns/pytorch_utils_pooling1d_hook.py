def pooling1d_hook(self, input, output):
    batch_size, input_channels, input_length = input[0].size()
    output_channels, output_length = output[0].size()
    kernel_ops = self.kernel_size[0]
    bias_ops = 0
    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_length
    list_pooling2d.append(flops)
