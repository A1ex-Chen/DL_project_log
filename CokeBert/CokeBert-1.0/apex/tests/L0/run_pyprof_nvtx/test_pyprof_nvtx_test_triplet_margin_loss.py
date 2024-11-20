def test_triplet_margin_loss(self):
    inp1 = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    inp2 = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    inp3 = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    output = F.triplet_margin_loss(inp1, inp2, inp3, margin=1.0, p=2, eps=
        1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
