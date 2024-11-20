def test_conv1d(self):
    tensor1d_in_conv = torch.randn(32, 3, 224, device='cuda', dtype=self.dtype)
    tensor1d_in_conv_grouped = torch.randn(32, 6, 224, device='cuda', dtype
        =self.dtype)
    conv1d_filter = torch.randn(16, 3, 3, device='cuda', dtype=self.dtype)
    conv1d_bias = torch.ones(16, device='cuda', dtype=self.dtype)
    conv1d_out_vanilla = F.conv1d(tensor1d_in_conv, conv1d_filter)
    conv1d_out_with_bias = F.conv1d(tensor1d_in_conv, conv1d_filter, bias=
        conv1d_bias)
    conv1d_out_strided = F.conv1d(tensor1d_in_conv, conv1d_filter, stride=2)
    conv1d_out_dilated = F.conv1d(tensor1d_in_conv, conv1d_filter, dilation=2)
    conv1d_out_grouped = F.conv1d(tensor1d_in_conv_grouped, conv1d_filter,
        groups=2)
    conv1d_out_padding_zeros = F.conv1d(tensor1d_in_conv, conv1d_filter,
        padding=6)
