def test_matmul_method_is_half(self):
    other = torch.randn(self.h, self.h)
    lhs = lambda x: x.matmul(other)
    rhs = lambda x: other.matmul(x)
    run_layer_test(self, [lhs, rhs], ALWAYS_HALF, (self.h, self.h))
