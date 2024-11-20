def fuse(self):
    if self.deploy:
        return
    LOGGER.info(f'fusing a RepConv block')
    self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
    self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
    rbr_1x1_bias = self.rbr_1x1.bias
    weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 
        1, 1, 1])
    if isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.
        rbr_identity, nn.modules.batchnorm.SyncBatchNorm):
        identity_conv_1x1 = nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.out_channels, kernel_size=1, stride=1,
            padding=0, groups=self.groups, bias=False)
        identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self
            .rbr_1x1.weight.data.device)
        identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze(
            ).squeeze()
        identity_conv_1x1.weight.data.fill_(0.0)
        identity_conv_1x1.weight.data.fill_diagonal_(1.0)
        identity_conv_1x1.weight.data = (identity_conv_1x1.weight.data.
            unsqueeze(2).unsqueeze(3))
        identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.
            rbr_identity)
        bias_identity_expanded = identity_conv_1x1.bias
        weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1
            .weight, [1, 1, 1, 1])
    else:
        bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(
            rbr_1x1_bias))
        weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(
            weight_1x1_expanded))
    self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight +
        weight_1x1_expanded + weight_identity_expanded)
    self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias +
        rbr_1x1_bias + bias_identity_expanded)
    self.rbr_reparam = self.rbr_dense
    self.deploy = True
    if self.rbr_identity is not None:
        del self.rbr_identity
        self.rbr_identity = None
    if self.rbr_1x1 is not None:
        del self.rbr_1x1
        self.rbr_1x1 = None
    if self.rbr_dense is not None:
        del self.rbr_dense
        self.rbr_dense = None
