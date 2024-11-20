def forward(self, x):
    x1 = self.linear(x)
    x1a = F.relu(x1)
    x1a = self.dropout(x1a)
    x2 = self.linear2(x1a)
    output = self.output(x2)
    return reshape(output, (-1,))
