def test_downsample_with_conv(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64)
    downsample = Downsample2D(channels=32, use_conv=True)
    with torch.no_grad():
        downsampled = downsample(sample)
    assert downsampled.shape == (1, 32, 32, 32)
    output_slice = downsampled[0, -1, -3:, -3:]
    expected_slice = torch.tensor([0.9267, 0.5878, 0.3337, 1.2321, -0.1191,
        -0.3984, -0.7532, -0.0715, -0.3913])
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
