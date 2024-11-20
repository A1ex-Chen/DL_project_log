def __init__(self, scope='train', criterion=nn.CrossEntropyLoss()):
    super().__init__('loss', scope=scope)
    self.criterion = criterion
