def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
    super(ESqueezeAndExciteFusionAdd, self).__init__()
    self.se_rgb = ESqueezeAndExcitation(channels_in, activation=activation)
    self.se_depth = ESqueezeAndExcitation(channels_in, activation=activation)
