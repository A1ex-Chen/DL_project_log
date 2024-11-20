def test_cosine_embedding_loss(self):
    inp1 = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    inp2 = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.randn(32, device='cuda', dtype=self.dtype, requires_grad
        =False)
    output = F.cosine_embedding_loss(inp1, inp2, target, margin=0,
        size_average=None, reduce=None, reduction='mean')
