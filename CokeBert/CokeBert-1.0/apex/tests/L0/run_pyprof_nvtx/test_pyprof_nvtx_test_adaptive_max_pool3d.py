def test_adaptive_max_pool3d(self):
    inp = torch.randn(1, 16, 16, 32, 32, device='cuda', dtype=self.dtype)
    out = F.adaptive_max_pool3d(inp, output_size=5, return_indices=True)
