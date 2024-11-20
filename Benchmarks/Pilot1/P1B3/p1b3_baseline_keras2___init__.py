def __init__(self, samples):
    super(MyProgbarLogger, self).__init__(count_mode='steps')
    self.samples = samples
    self.params = {}
