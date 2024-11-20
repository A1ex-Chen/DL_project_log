def test_smooth_l1_loss(self):
    inp = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=False)
    output = F.smooth_l1_loss(inp, target, size_average=None, reduce=None,
        reduction='mean')
