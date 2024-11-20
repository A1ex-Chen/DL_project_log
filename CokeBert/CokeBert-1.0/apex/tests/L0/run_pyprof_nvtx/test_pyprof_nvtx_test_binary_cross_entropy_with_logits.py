def test_binary_cross_entropy_with_logits(self):
    inp = torch.randn(32, 128, device='cuda', dtype=self.dtype,
        requires_grad=True)
    target = torch.empty_like(inp).random_(2)
    output = F.binary_cross_entropy_with_logits(inp, target)
