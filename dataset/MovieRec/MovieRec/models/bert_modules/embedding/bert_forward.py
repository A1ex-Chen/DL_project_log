def forward(self, sequence):
    x = self.token(sequence) + self.position(sequence)
    return self.dropout(x)
