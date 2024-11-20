def fuse_deconv_and_bn(deconv, bn):
    """Fuse ConvTranspose2d() and BatchNorm2d() layers."""
    fuseddconv = nn.ConvTranspose2d(deconv.in_channels, deconv.out_channels,
        kernel_size=deconv.kernel_size, stride=deconv.stride, padding=
        deconv.padding, output_padding=deconv.output_padding, dilation=
        deconv.dilation, groups=deconv.groups, bias=True).requires_grad_(False
        ).to(deconv.weight.device)
    w_deconv = deconv.weight.clone().view(deconv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fuseddconv.weight.copy_(torch.mm(w_bn, w_deconv).view(fuseddconv.weight
        .shape))
    b_conv = torch.zeros(deconv.weight.shape[1], device=deconv.weight.device
        ) if deconv.bias is None else deconv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.
        running_var + bn.eps))
    fuseddconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) +
        b_bn)
    return fuseddconv
