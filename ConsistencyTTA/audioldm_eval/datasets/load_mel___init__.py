def __init__(self, datadir, sr=16000, target_length=1000, limit_num=None):
    self.datalist = [os.path.join(datadir, x) for x in os.listdir(datadir)]
    self.datalist = sorted(self.datalist)
    self.datalist = [item for item in self.datalist if item.endswith('.wav')]
    if limit_num is not None:
        self.datalist = self.datalist[:limit_num]
    self.sr = sr
    self.target_length = target_length
