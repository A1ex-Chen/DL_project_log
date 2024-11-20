def BilinearInterpolation(tensor_in, up_scale):
    assert up_scale % 2 == 0, 'Scale should be even'

    def upsample_filt(size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center
            ) / factor)
    kernel_size = int(up_scale) * 2
    bil_filt = upsample_filt(kernel_size)
    dim = int(tensor_in.shape[1])
    kernel = np.zeros((dim, dim, kernel_size, kernel_size), dtype=np.float32)
    kernel[range(dim), range(dim), :, :] = bil_filt
    tensor_out = F.conv_transpose2d(tensor_in, weight=to_device(torch.
        Tensor(kernel), tensor_in.device), bias=None, stride=int(up_scale),
        padding=int(up_scale / 2))
    return tensor_out
