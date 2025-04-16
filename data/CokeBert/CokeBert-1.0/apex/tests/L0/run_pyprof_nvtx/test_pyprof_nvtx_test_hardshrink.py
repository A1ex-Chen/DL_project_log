def test_hardshrink(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.hardshrink(inp, lambd=0.5)
