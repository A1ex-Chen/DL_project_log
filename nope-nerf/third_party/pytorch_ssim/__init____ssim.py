def _ssim(img1, img2, window, window_size, channel, use_padding,
    size_average=True):
    if use_padding:
        padding_size = window_size // 2
    else:
        padding_size = 0
    mu1 = F.conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding_size, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding_size, groups=
        channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding_size, groups=
        channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding_size, groups=
        channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2) / ((mu1_sq + mu2_sq +
        C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
