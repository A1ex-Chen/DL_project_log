def test_softplus(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.softplus(inp, beta=1, threshold=20)
