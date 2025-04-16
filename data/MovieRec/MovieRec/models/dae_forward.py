def forward(self, x):
    x = F.normalize(x)
    x = self.input_dropout(x)
    for i, layer in enumerate(self.encoder):
        x = layer(x)
        x = torch.tanh(x)
    for i, layer in enumerate(self.decoder):
        x = layer(x)
        if i != len(self.decoder) - 1:
            x = torch.tanh(x)
    return x
