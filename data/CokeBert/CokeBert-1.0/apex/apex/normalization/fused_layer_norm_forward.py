def forward(self, input):
    if not input.is_cuda:
        return F.layer_norm(input, self.normalized_shape, self.weight, self
            .bias, self.eps)
    if self.elementwise_affine:
        return FusedLayerNormAffineFunction.apply(input, self.weight, self.
            bias, self.normalized_shape, self.eps)
    else:
        return FusedLayerNormFunction.apply(input, self.normalized_shape,
            self.eps)
