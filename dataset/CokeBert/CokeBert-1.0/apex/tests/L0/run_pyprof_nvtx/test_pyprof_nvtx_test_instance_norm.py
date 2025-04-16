def test_instance_norm(self):
    inp = torch.randn(1, 3, 32, 32, device='cuda', dtype=self.dtype)
    running_mean = torch.randn(3, device='cuda', dtype=self.dtype)
    running_var = torch.randn(3, device='cuda', dtype=self.dtype)
    output = F.instance_norm(inp, running_mean=running_mean, running_var=
        running_var, weight=None, bias=None, use_input_stats=True, momentum
        =0.1, eps=1e-05)
