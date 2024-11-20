def test_mse_loss(self):
    inp = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    output = F.mse_loss(inp, target, size_average=None, reduce=None,
        reduction='mean')
