def forward(self, data, target):
    output = self.model(data)
    loss = self.loss(output, target)
    return loss, output
