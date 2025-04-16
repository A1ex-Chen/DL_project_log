def fre_init(self):
    prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.
        kernel_size)
    half_fg = self.out_channels / 2
    for i in range(self.out_channels):
        for h in range(3):
            for w in range(3):
                if i < half_fg:
                    prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) *
                        (i + 1) / 3)
                else:
                    prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) *
                        (i + 1 - half_fg) / 3)
    self.register_buffer('weight_rbr_prior', prior_tensor)
