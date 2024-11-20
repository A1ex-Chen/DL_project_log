def forward(self, x):
    """Executes feedforward within the neural network module and applies activation."""
    for i, layer in enumerate(self.layers):
        x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
    if self.sigmoid_output:
        x = torch.sigmoid(x)
    return x
