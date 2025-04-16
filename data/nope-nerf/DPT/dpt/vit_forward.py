def forward(self, x):
    x = x.transpose(self.dim0, self.dim1)
    return x
