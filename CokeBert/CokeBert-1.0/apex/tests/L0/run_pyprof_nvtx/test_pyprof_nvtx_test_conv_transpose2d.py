def test_conv_transpose2d(self):
    conv_transpose2d_tensor = torch.randn(64, 8, 5, 5, device='cuda', dtype
        =self.dtype)
    conv_transpose2d_filter = torch.randn(8, 16, 3, 3, device='cuda', dtype
        =self.dtype)
    conv_transpose2d_bias = torch.randn(16, device='cuda', dtype=self.dtype)
    conv_transpose2d_out = F.conv_transpose2d(conv_transpose2d_tensor,
        conv_transpose2d_filter)
    conv_transpose2d_out_biased = F.conv_transpose2d(conv_transpose2d_tensor,
        conv_transpose2d_filter, bias=conv_transpose2d_bias)
    conv_transpose2d_out_strided = F.conv_transpose2d(conv_transpose2d_tensor,
        conv_transpose2d_filter, stride=2)
    conv_transpose2d_out_padded = F.conv_transpose2d(conv_transpose2d_tensor,
        conv_transpose2d_filter, padding=3)
    conv_transpose2d_out2_padded = F.conv_transpose2d(conv_transpose2d_tensor,
        conv_transpose2d_filter, output_padding=2, dilation=3)
    conv_transpose2d_out_grouped = F.conv_transpose2d(conv_transpose2d_tensor,
        conv_transpose2d_filter, groups=2)
    conv_transpose2d_out_dilated = F.conv_transpose2d(conv_transpose2d_tensor,
        conv_transpose2d_filter, dilation=2)
