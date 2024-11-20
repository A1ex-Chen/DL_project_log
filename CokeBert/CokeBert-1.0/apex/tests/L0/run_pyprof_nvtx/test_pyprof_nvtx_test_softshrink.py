def test_softshrink(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.softshrink(inp, lambd=0.5)
