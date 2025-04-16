def setUp(self):
    common_init(self)
    self.val = 4.0
    self.overflow_buf = torch.cuda.IntTensor(1).zero_()
