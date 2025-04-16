def test_conv2d(self):
    tensor2d_in_conv = torch.randn(32, 3, 224, 224, device='cuda', dtype=
        self.dtype)
    tensor2d_in_conv_grouped = torch.randn(32, 6, 224, 224, device='cuda',
        dtype=self.dtype)
    conv2d_filter = torch.randn(16, 3, 3, 3, device='cuda', dtype=self.dtype)
    conv2d_bias = torch.ones(16, device='cuda', dtype=self.dtype)
    conv2d_out_vanilla = F.conv2d(tensor2d_in_conv, conv2d_filter)
    conv2d_with_bias = F.conv2d(tensor2d_in_conv, conv2d_filter, bias=
        conv2d_bias)
    conv2d_out_strided = F.conv2d(tensor2d_in_conv, conv2d_filter, stride=2)
    conv2d_out_dilated = F.conv2d(tensor2d_in_conv, conv2d_filter, dilation=2)
    conv2d_out_grouped = F.conv2d(tensor2d_in_conv_grouped, conv2d_filter,
        groups=2)
    conv2d_out_padding_zeros = F.conv2d(tensor2d_in_conv, conv2d_filter,
        padding=6)
