def test_lp_pool1d(self):
    inp = torch.randn(1, 32, 64, device='cuda', dtype=self.dtype)
    output = F.lp_pool1d(inp, 2, 3, stride=2, ceil_mode=True)
