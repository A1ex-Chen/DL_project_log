def __init__(self, in_channels):
    super(InceptionE_2, self).__init__()
    self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)
    self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
    self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(
        0, 1))
    self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(
        1, 0))
    self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
    self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
    self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3),
        padding=(0, 1))
    self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1),
        padding=(1, 0))
    self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)
