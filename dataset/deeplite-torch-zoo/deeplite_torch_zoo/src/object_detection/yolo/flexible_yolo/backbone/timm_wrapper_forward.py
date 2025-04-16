def forward(self, x):
    outs = self.backbone(x)
    if isinstance(outs, (list, tuple)):
        features = tuple(outs)
    else:
        features = outs,
    return tuple(features[idx] for idx in self.feature_map_indices)
