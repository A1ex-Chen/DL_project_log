def test_pdist(self):
    inp = torch.randn(128, 128, device='cuda', dtype=torch.float32)
    output = F.pdist(inp, p=2)
