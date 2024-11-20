def gaussian_filter(latents, kernel_size=3, sigma=1.0):
    channels = latents.shape[1]
    kernel = gaussian_kernel(kernel_size, sigma, channels).to(latents.
        device, latents.dtype)
    blurred_latents = F.conv2d(latents, kernel, padding=kernel_size // 2,
        groups=channels)
    return blurred_latents
