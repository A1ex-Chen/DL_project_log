def test_linear_is_half(self):
    m = nn.Linear(self.h, self.h)
    f = ft.partial(F.linear, weight=m.weight, bias=m.bias)
    run_layer_test(self, [m, f], ALWAYS_HALF, (self.b, self.h))
