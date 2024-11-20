def test_threshold_(self):
    inp = torch.randn(1, 8, 32, 32, device='cuda', dtype=self.dtype)
    output = F.threshold_(inp, 6, 6)
