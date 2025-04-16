def test_softmax_is_float(self):
    m = nn.Softmax(dim=1)
    f = ft.partial(F.softmax, dim=1)
    run_layer_test(self, [m, f], ALWAYS_FLOAT, (self.b, self.h))
