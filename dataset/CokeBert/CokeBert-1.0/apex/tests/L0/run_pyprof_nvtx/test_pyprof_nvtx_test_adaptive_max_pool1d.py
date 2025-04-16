def test_adaptive_max_pool1d(self):
    inp = torch.randn(1, 16, 28, device='cuda', dtype=self.dtype)
    out = F.adaptive_max_pool1d(inp, output_size=5, return_indices=True)
