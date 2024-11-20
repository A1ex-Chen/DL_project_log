def test_cosine_similarity(self):
    inp1 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
    inp2 = torch.randn(1024, 128, device='cuda', dtype=self.dtype)
    output = F.cosine_similarity(inp1, inp2, dim=1, eps=1e-08)
