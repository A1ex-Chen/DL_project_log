def forward_slicing(self, x: Tensor) ->Tensor:
    x = x.clone()
    x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3,
        :, :])
    return x
