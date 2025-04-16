def setUp(self):
    self.x = torch.ones(2, device='cuda', dtype=torch.float32)
    common_init(self)
