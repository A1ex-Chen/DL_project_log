def fisher_forward_linear(self, x):
    x = nn.functional.linear(x, self.weight, self.bias)
    self.act = self.dummy(x)
    return self.act
