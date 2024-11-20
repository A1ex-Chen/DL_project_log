def test_upsample_with_conv_out_dim(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 32, 32)
    upsample = Upsample2D(channels=32, use_conv=True, out_channels=64)
    with torch.no_grad():
        upsampled = upsample(sample)
    assert upsampled.shape == (1, 64, 64, 64)
    output_slice = upsampled[0, -1, -3:, -3:]
    expected_slice = torch.tensor([0.2703, 0.1656, -0.2538, -0.0553, -
        0.2984, 0.1044, 0.1155, 0.2579, 0.7755])
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
