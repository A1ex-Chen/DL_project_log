def test_logsigmoid(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.logsigmoid(inp)
