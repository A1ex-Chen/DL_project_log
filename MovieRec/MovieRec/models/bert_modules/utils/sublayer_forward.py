def forward(self, x, sublayer):
    """Apply residual connection to any sublayer with the same size."""
    return x + self.dropout(sublayer(self.norm(x)))
