def setUp(self):
    self.N = 64
    self.C_in = 3
    self.H_in = 16
    self.W_in = 32
    self.in_tensor = torch.randn((self.N, self.C_in, self.H_in, self.W_in)
        ).cuda()
    self.orig_model = DummyNetWrapper().cuda()
    self.fp16_model = FP16Model(self.orig_model)
