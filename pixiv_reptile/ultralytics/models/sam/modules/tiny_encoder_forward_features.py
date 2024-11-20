def forward_features(self, x):
    """Runs the input through the model layers and returns the transformed output."""
    x = self.patch_embed(x)
    x = self.layers[0](x)
    start_i = 1
    for i in range(start_i, len(self.layers)):
        layer = self.layers[i]
        x = layer(x)
    batch, _, channel = x.shape
    x = x.view(batch, 64, 64, channel)
    x = x.permute(0, 3, 1, 2)
    return self.neck(x)
