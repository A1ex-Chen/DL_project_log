def test_batch_norm_is_match(self):
    m = nn.BatchNorm2d(num_features=self.c)
    f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m
        .running_var, weight=m.weight, bias=m.bias, training=True)
    run_layer_test(self, [m], MATCH_INPUT, (self.b, self.c, self.h, self.h))
    m.eval()
    f = ft.partial(F.batch_norm, running_mean=m.running_mean, running_var=m
        .running_var, weight=m.weight, bias=m.bias, training=False)
    run_layer_test(self, [m, f], MATCH_INPUT, (self.b, self.c, self.h, self
        .h), test_backward=False)
