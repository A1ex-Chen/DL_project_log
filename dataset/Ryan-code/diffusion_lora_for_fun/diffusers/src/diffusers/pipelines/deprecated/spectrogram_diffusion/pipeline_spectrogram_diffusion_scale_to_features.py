def scale_to_features(self, outputs, input_range=(-1.0, 1.0), clip=False):
    """Invert by linearly scaling network outputs to features range."""
    min_out, max_out = input_range
    outputs = torch.clip(outputs, min_out, max_out) if clip else outputs
    zero_one = (outputs - min_out) / (max_out - min_out)
    return zero_one * (self.max_value - self.min_value) + self.min_value
