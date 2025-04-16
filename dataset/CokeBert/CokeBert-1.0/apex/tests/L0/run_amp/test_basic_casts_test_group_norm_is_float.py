def test_group_norm_is_float(self):
    m = nn.GroupNorm(num_groups=4, num_channels=self.c)
    run_layer_test(self, [m], ALWAYS_FLOAT, (self.b, self.c, self.h, self.h))
