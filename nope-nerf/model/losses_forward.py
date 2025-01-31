def forward(self, x, y):
    x = self.refl(x)
    y = self.refl(y)
    mu_x = self.mu_x_pool(x)
    mu_y = self.mu_y_pool(y)
    sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
    sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
    sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
