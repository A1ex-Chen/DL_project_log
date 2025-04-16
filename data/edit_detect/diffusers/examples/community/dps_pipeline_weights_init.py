def weights_init(self):
    if self.blur_type == 'gaussian':
        n = np.zeros((self.kernel_size, self.kernel_size))
        n[self.kernel_size // 2, self.kernel_size // 2] = 1
        k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
        k = torch.from_numpy(k)
        self.k = k
        for name, f in self.named_parameters():
            f.data.copy_(k)
