def forward(self, x: torch.Tensor):
    out = {}
    start = 0
    for k, shape in zip(self.config.param_names, self.config.param_shapes):
        vectors, _ = shape
        end = start + vectors
        x_bvd = x[:, start:end]
        out[k] = self.projections[_sanitize_name(k)](x_bvd).reshape(len(x),
            *shape)
        start = end
    return out
