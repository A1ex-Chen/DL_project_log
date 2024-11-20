@staticmethod
def _output_size(input, weight, padding, dilation, stride):
    channels = weight.size(0)
    output_size = input.size(0), channels
    for d in range(input.dim() - 2):
        in_size = input.size(d + 2)
        pad = padding[d]
        kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
        stride_ = stride[d]
        output_size += (in_size + 2 * pad - kernel) // stride_ + 1,
    if not all(map(lambda s: s > 0, output_size)):
        raise ValueError('convolution input is too small (output would be {})'
            .format('x'.join(map(str, output_size))))
    return output_size
