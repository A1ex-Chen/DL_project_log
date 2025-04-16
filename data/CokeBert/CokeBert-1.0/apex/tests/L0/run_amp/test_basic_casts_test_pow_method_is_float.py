def test_pow_method_is_float(self):
    fn = lambda x: x.pow(2.0)
    run_layer_test(self, [fn], ALWAYS_FLOAT, (self.b, self.h))
