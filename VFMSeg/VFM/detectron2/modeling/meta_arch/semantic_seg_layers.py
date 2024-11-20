def layers(self, features):
    for i, f in enumerate(self.in_features):
        if i == 0:
            x = self.scale_heads[i](features[f])
        else:
            x = x + self.scale_heads[i](features[f])
    x = self.predictor(x)
    return x
