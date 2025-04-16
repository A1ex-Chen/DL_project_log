def __init__(self, channels, groups=32, eps=1e-05, nonlinear=True):
    super(EvoNorm2dS0, self).__init__()
    self.groups = groups
    self.eps = eps
    self.nonlinear = nonlinear
    self.gamma = self.add_weight(name='gamma', shape=(1, 1, 1, channels),
        initializer=tf.initializers.Ones())
    self.beta = self.add_weight(name='beta', shape=(1, 1, 1, channels),
        initializer=tf.initializers.Zeros())
    self.v = self.add_weight(name='v', shape=(1, 1, 1, channels),
        initializer=tf.initializers.Ones())
