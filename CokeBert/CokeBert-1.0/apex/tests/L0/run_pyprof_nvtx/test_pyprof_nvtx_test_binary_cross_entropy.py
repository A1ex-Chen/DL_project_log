def test_binary_cross_entropy(self):
    inp = torch.randn(32, 128, device='cuda', dtype=torch.float32,
        requires_grad=True)
    target = torch.randn(32, 128, device='cuda', dtype=torch.float32,
        requires_grad=False)
    output = F.binary_cross_entropy(torch.sigmoid(inp), target)
