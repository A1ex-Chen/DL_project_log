def test_matmul_op_is_half(self):
    other = torch.randn(self.h, self.h)
    lhs = lambda x: x @ other
    rhs = lambda x: other @ x
    run_layer_test(self, [lhs, rhs], ALWAYS_HALF, (self.h, self.h))
