def forward(self, x):
    return self.activation(self.block(x) + x)
