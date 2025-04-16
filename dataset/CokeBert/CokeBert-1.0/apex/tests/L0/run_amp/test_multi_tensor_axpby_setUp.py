def setUp(self):
    common_init(self)
    self.a = 2.0
    self.b = 8.0
    self.xval = 4.0
    self.yval = 16.0
    self.overflow_buf = torch.cuda.IntTensor(1).zero_()
    self.ref = torch.cuda.FloatTensor([136.0])
