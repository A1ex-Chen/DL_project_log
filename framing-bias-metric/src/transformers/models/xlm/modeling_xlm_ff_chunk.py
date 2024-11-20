def ff_chunk(self, input):
    x = self.lin1(input)
    x = self.act(x)
    x = self.lin2(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    return x
