def test_adaptive_avg_pool2d(self):
    inp = torch.randn(1, 16, 32, 32, device='cuda', dtype=self.dtype)
    out = F.adaptive_avg_pool2d(inp, output_size=5)
