def forward(self, img1, img2):
    _, channel, _, _ = img1.size()
    if channel == self.channel and self.window.data.type() == img1.data.type():
        window = self.window
    else:
        window = create_window(self.window_size, channel)
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        self.window = window
        self.channel = channel
    return _ssim(img1, img2, window, self.window_size, channel, self.
        use_padding, self.size_average)
