def test_mse_loss_is_float(self):
    shape = self.b, self.h
    target = torch.randn(shape)
    mod = nn.MSELoss()
    m = lambda x: mod(x, target)
    f = ft.partial(F.mse_loss, target=target)
    run_layer_test(self, [m], ALWAYS_FLOAT, shape)
