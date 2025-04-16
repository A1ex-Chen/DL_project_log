def test_sum_is_float(self):
    fn = lambda x: x.sum()
    run_layer_test(self, [fn], ALWAYS_FLOAT, (self.b, self.h))
