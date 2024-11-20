def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0):
    super().__init__()
    self.blur_type = blur_type
    self.kernel_size = kernel_size
    self.std = std
    self.seq = nn.Sequential(nn.ReflectionPad2d(self.kernel_size // 2), nn.
        Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False,
        groups=3))
    self.weights_init()
