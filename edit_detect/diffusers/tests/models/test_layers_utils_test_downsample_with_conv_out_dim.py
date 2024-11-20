def test_downsample_with_conv_out_dim(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64)
    downsample = Downsample2D(channels=32, use_conv=True, out_channels=16)
    with torch.no_grad():
        downsampled = downsample(sample)
    assert downsampled.shape == (1, 16, 32, 32)
    output_slice = downsampled[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-0.6586, 0.5985, 0.0721, 0.1256, -0.1492,
        0.4436, -0.2544, 0.5021, 1.1522])
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
