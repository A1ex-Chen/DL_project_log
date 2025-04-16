def __init__(self, channels, groups):
    super(ChannelShuffle, self).__init__()
    if channels % groups != 0:
        raise ValueError('channels must be divisible by groups')
    self.groups = groups
