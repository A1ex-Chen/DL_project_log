def __init__(self, drop_prob=None):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob
