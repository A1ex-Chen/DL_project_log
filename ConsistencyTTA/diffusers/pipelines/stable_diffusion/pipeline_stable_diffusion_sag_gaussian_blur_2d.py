def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5
    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)
    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0],
        kernel2d.shape[1])
    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, 
        kernel_size // 2]
    img = F.pad(img, padding, mode='reflect')
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img
