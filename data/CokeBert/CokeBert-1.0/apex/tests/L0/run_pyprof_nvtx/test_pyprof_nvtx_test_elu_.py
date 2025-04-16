def test_elu_(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.elu_(inp, alpha=1.0)
