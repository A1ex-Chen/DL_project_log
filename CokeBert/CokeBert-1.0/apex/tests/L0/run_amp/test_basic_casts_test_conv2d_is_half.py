def test_conv2d_is_half(self):
    m = nn.Conv2d(self.c, self.c, self.k)
    f = ft.partial(F.conv2d, weight=m.weight, bias=m.bias)
    run_layer_test(self, [m, f], ALWAYS_HALF, (self.b, self.c, self.h, self.h))
