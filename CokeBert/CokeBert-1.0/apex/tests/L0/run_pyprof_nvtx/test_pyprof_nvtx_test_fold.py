def test_fold(self):
    inp = torch.randn(3, 20, 20, device='cuda', dtype=self.dtype)
    inp_folded = F.fold(inp, (4, 5), (1, 1))
