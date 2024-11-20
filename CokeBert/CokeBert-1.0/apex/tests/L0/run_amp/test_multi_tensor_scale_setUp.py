def setUp(self):
    common_init(self)
    self.scale = 4.0
    self.overflow_buf = torch.cuda.IntTensor(1).zero_()
    self.ref = torch.cuda.FloatTensor([1.0])
