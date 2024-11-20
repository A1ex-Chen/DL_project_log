def _ssim(self, img1, img2, size_average=True):
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11
    window = create_window(window_size, sigma, self.channel).cuda(img1.
        get_device())
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=self.channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=self.channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2,
        groups=self.channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2,
        groups=self.channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2,
        groups=self.channel) - mu1_mu2
    C1 = (0.01 * self.max_val) ** 2
    C2 = (0.03 * self.max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = (2 * mu1_mu2 + C1) * V1 / ((mu1_sq + mu2_sq + C1) * V2)
    mcs_map = V1 / V2
    if size_average:
        return ssim_map.mean(), mcs_map.mean()
