def forward(self, x):
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.variance_epsilon)
    return self.weight * x + self.bias
