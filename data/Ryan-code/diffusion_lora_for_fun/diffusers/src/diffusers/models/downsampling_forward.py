def forward(self, inputs: torch.Tensor) ->torch.Tensor:
    inputs = F.pad(inputs, (self.pad,) * 4, self.pad_mode)
    weight = inputs.new_zeros([inputs.shape[1], inputs.shape[1], self.
        kernel.shape[0], self.kernel.shape[1]])
    indices = torch.arange(inputs.shape[1], device=inputs.device)
    kernel = self.kernel.to(weight)[None, :].expand(inputs.shape[1], -1, -1)
    weight[indices, indices] = kernel
    return F.conv2d(inputs, weight, stride=2)
