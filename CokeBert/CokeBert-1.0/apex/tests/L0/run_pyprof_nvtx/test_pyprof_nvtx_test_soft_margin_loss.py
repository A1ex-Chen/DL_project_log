def test_soft_margin_loss(self):
    inp = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=False)
    output = F.soft_margin_loss(inp, target, size_average=None, reduce=None,
        reduction='mean')
