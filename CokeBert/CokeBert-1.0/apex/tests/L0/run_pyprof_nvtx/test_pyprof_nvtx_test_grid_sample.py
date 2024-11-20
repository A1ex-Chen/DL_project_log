def test_grid_sample(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    grid = torch.randn(16, 32, 32, 2, device='cuda', dtype=self.dtype)
    output = F.grid_sample(inp, grid, mode='bilinear', padding_mode='zeros')
