def test_adaptive_avg_pool1d(self):
    inp = torch.randn(1, 1, 28, device='cuda', dtype=self.dtype)
    out = F.adaptive_avg_pool1d(inp, output_size=5)
