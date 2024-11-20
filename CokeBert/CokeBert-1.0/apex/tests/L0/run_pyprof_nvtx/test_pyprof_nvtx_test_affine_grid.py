def test_affine_grid(self):
    theta = torch.randn(32, 2, 3, device='cuda', dtype=self.dtype)
    size = 32, 8, 32, 32
    output = F.affine_grid(theta, size)
