def test_conv_transpose3d(self):
    conv_transpose3d_tensor = torch.randn(20, 16, 50, 10, 20, device='cuda',
        dtype=self.dtype)
    conv_transpose3d_filter = torch.randn(16, 33, 3, 3, 3, device='cuda',
        dtype=self.dtype)
    conv_transpose3d_bias = torch.randn(33, device='cuda', dtype=self.dtype)
    conv_transpose3d_out = F.conv_transpose3d(conv_transpose3d_tensor,
        conv_transpose3d_filter)
    conv_transpose3d_out_biased = F.conv_transpose3d(conv_transpose3d_tensor,
        conv_transpose3d_filter, bias=conv_transpose3d_bias)
    conv_transpose3d_out_strided = F.conv_transpose3d(conv_transpose3d_tensor,
        conv_transpose3d_filter, stride=2)
    conv_transpose3d_out_padded = F.conv_transpose3d(conv_transpose3d_tensor,
        conv_transpose3d_filter, padding=3)
    conv_transpose3d_out2_padded = F.conv_transpose3d(conv_transpose3d_tensor,
        conv_transpose3d_filter, output_padding=2, dilation=3)
    conv_transpose3d_out_grouped = F.conv_transpose3d(conv_transpose3d_tensor,
        conv_transpose3d_filter, groups=2)
    conv_transpose3d_out_dilated = F.conv_transpose3d(conv_transpose3d_tensor,
        conv_transpose3d_filter, dilation=2)
