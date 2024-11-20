def test_lp_pool2d(self):
    inp = torch.randn(1, 32, 64, 64, device='cuda', dtype=self.dtype)
    output = F.lp_pool2d(inp, 2, 3, stride=2, ceil_mode=True)
