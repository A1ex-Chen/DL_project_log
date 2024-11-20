def ff_chunk(self, input):
    x = self.lin1(input)
    x = self.activation(x)
    x = self.lin2(x)
    x = self.dropout(x)
    return x
