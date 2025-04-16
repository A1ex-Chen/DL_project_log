def _test_same_output(self, batch_size):
    torch.cuda.manual_seed(42)
    self.input_ = torch.randn((batch_size, *self.module_cpu_.
        normalized_shape), device='cpu').requires_grad_(True)
    self.input_cuda_ = self.input_.cuda().detach().requires_grad_(True)
    out_cpu_ = self.module_cpu_(self.input_)
    gO = torch.rand_like(out_cpu_)
    out_cpu_.backward(gO)
    out_cuda_ = self.module_cuda_(self.input_cuda_)
    gO = gO.cuda()
    out_cuda_.backward(gO)
    assert out_cpu_.is_cuda == False
    assert out_cuda_.is_cuda == True
    torch.testing.assert_allclose(out_cpu_, out_cuda_.cpu())
    torch.testing.assert_allclose(self.input_.grad, self.input_cuda_.grad.cpu()
        )
