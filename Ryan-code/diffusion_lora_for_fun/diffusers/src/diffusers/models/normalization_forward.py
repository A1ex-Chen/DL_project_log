def forward(self, input):
    return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)
