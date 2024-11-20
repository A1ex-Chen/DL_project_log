@staticmethod
def _infer_shape(ctx, input, weight):
    n = input.size(0)
    channels_out = weight.size(0)
    height, width = input.shape[2:4]
    kernel_h, kernel_w = weight.shape[2:4]
    height_out = (height + 2 * ctx.padding - (ctx.dilation * (kernel_h - 1) +
        1)) // ctx.stride + 1
    width_out = (width + 2 * ctx.padding - (ctx.dilation * (kernel_w - 1) + 1)
        ) // ctx.stride + 1
    return n, channels_out, height_out, width_out
