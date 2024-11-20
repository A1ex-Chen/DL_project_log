def __init__(self, window_size=11, use_padding=True, size_average=True):
    super(SSIM, self).__init__()
    self.window_size = window_size
    self.size_average = size_average
    self.use_padding = use_padding
    self.channel = 1
    self.window = create_window(window_size, self.channel)
