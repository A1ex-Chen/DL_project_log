def forward(self, x):
    if not torch.jit.is_scripting():
        if x.numel() == 0 and self.training:
            assert not isinstance(self.norm, torch.nn.SyncBatchNorm
                ), 'SyncBatchNorm does not support empty inputs!'
    x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self
        .dilation, self.groups)
    if self.norm is not None:
        x = self.norm(x)
    if self.activation is not None:
        x = self.activation(x)
    return x
