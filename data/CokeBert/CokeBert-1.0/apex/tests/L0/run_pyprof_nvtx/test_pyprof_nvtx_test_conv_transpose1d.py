def test_conv_transpose1d(self):
    conv_transpose1d_tensor = torch.randn(64, 16, 64, device='cuda', dtype=
        self.dtype)
    conv_transpose1d_filter = torch.randn(16, 32, 3, device='cuda', dtype=
        self.dtype)
    conv_transpose1d_bias = torch.randn(32, device='cuda', dtype=self.dtype)
    conv_transpose1d_out = F.conv_transpose1d(conv_transpose1d_tensor,
        conv_transpose1d_filter)
    conv_transpose1d_out_biased = F.conv_transpose1d(conv_transpose1d_tensor,
        conv_transpose1d_filter, bias=conv_transpose1d_bias)
    conv_transpose1d_out_strided = F.conv_transpose1d(conv_transpose1d_tensor,
        conv_transpose1d_filter, stride=2)
    conv_transpose1d_out_padded = F.conv_transpose1d(conv_transpose1d_tensor,
        conv_transpose1d_filter, padding=3)
    conv_transpose1d_out2_padded = F.conv_transpose1d(conv_transpose1d_tensor,
        conv_transpose1d_filter, output_padding=2, dilation=3)
    conv_transpose1d_out_grouped = F.conv_transpose1d(conv_transpose1d_tensor,
        conv_transpose1d_filter, groups=2)
    conv_transpose1d_out_dilated = F.conv_transpose1d(conv_transpose1d_tensor,
        conv_transpose1d_filter, dilation=2)
