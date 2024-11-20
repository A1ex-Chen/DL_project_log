def forward(self, x, offset, mask):
    if x.numel() == 0:
        output_shape = [((i + 2 * p - (di * (k - 1) + 1)) // s + 1) for i,
            p, di, k, s in zip(x.shape[-2:], self.padding, self.dilation,
            self.kernel_size, self.stride)]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
    x = modulated_deform_conv(x, offset, mask, self.weight, self.bias, self
        .stride, self.padding, self.dilation, self.groups, self.
        deformable_groups)
    if self.norm is not None:
        x = self.norm(x)
    if self.activation is not None:
        x = self.activation(x)
    return x
