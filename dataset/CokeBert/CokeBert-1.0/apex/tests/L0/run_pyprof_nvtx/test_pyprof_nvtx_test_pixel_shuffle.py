def test_pixel_shuffle(self):
    inp = torch.randn(16, 8, 64, 64, device='cuda', dtype=self.dtype)
    output = torch.nn.functional.pixel_shuffle(inp, 2)
