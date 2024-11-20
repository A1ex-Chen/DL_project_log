def test_layer_norm(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    output = F.layer_norm(inp, inp.size()[1:], weight=None, bias=None, eps=
        1e-05)
