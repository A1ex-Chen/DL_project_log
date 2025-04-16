def scale_features(self, features, output_range=(-1.0, 1.0), clip=False):
    """Linearly scale features to network outputs range."""
    min_out, max_out = output_range
    if clip:
        features = torch.clip(features, self.min_value, self.max_value)
    zero_one = (features - self.min_value) / (self.max_value - self.min_value)
    return zero_one * (max_out - min_out) + min_out
