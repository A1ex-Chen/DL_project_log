def test_conv3d(self):
    tensor3d_in_conv = torch.randn(32, 3, 16, 224, 224, device='cuda',
        dtype=self.dtype)
    tensor3d_in_conv_grouped = torch.randn(32, 6, 16, 224, 224, device=
        'cuda', dtype=self.dtype)
    conv3d_filter = torch.randn(16, 3, 3, 3, 3, device='cuda', dtype=self.dtype
        )
    conv3d_bias = torch.ones(16, device='cuda', dtype=self.dtype)
    conv3d_out_vanilla = F.conv3d(tensor3d_in_conv, conv3d_filter)
    conv3d_out_strided = F.conv3d(tensor3d_in_conv, conv3d_filter, stride=2)
    conv3d_out_dilated = F.conv3d(tensor3d_in_conv, conv3d_filter, dilation=2)
    conv3d_out_grouped = F.conv3d(tensor3d_in_conv_grouped, conv3d_filter,
        groups=2)
    conv3d_out_padding_zeros = F.conv3d(tensor3d_in_conv, conv3d_filter,
        padding=6)
