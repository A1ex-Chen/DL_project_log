def test_downsample_default(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64)
    downsample = Downsample2D(channels=32, use_conv=False)
    with torch.no_grad():
        downsampled = downsample(sample)
    assert downsampled.shape == (1, 32, 32, 32)
    output_slice = downsampled[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-0.0513, -0.3889, 0.064, 0.0836, -0.546,
        -0.0341, -0.0169, -0.6967, 0.1179])
    max_diff = (output_slice.flatten() - expected_slice).abs().sum().item()
    assert max_diff <= 0.001
