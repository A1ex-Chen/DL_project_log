def __init__(self, h):
    super(Generator, self).__init__()
    self.h = h
    self.num_kernels = len(h.resblock_kernel_sizes)
    self.num_upsamples = len(h.upsample_rates)
    self.conv_pre = weight_norm(Conv1d(h.num_mels, h.
        upsample_initial_channel, 7, 1, padding=3))
    resblock = ResBlock
    self.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
        self.ups.append(weight_norm(ConvTranspose1d(h.
            upsample_initial_channel // 2 ** i, h.upsample_initial_channel //
            2 ** (i + 1), k, u, padding=(k - u) // 2)))
    self.resblocks = nn.ModuleList()
    for i in range(len(self.ups)):
        ch = h.upsample_initial_channel // 2 ** (i + 1)
        for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
            self.resblocks.append(resblock(h, ch, k, d))
    self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
    self.ups.apply(init_weights)
    self.conv_post.apply(init_weights)
