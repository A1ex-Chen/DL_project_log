def test_pow_op_is_float(self):
    fn = lambda x: x ** 2.0
    run_layer_test(self, [fn], ALWAYS_FLOAT, (self.b, self.h))
