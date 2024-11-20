def forward(self, x):
    batch_size = x.size(0)
    return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
