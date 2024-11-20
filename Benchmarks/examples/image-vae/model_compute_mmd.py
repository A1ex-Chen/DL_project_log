def compute_mmd(self, x, y):
    x_kernel = self.compute_kernel(x, x)
    y_kernel = self.compute_kernel(y, y)
    xy_kernel = self.compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(
        xy_kernel)
