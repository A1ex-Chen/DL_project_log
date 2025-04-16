def test_relu_is_match(self):
    run_layer_test(self, [nn.ReLU(), F.relu], MATCH_INPUT, (self.b, self.h))
