def test_cross_entropy(self):
    inp = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randint(0, 100, (32,), device='cuda', dtype=torch.long,
        requires_grad=False)
    output = F.cross_entropy(inp, target, weight=None, size_average=None,
        ignore_index=-100, reduce=None, reduction='mean')
