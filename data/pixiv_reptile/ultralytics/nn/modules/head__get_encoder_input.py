def _get_encoder_input(self, x):
    """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
    x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
    feats = []
    shapes = []
    for feat in x:
        h, w = feat.shape[2:]
        feats.append(feat.flatten(2).permute(0, 2, 1))
        shapes.append([h, w])
    feats = torch.cat(feats, 1)
    return feats, shapes
