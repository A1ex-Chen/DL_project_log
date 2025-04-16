def test_hardtanh(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.hardtanh(inp, min_val=-1.0, max_val=1.0, inplace=False)
