def test_upsample_with_conv(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 32, 32)
    upsample = Upsample2D(channels=32, use_conv=True)
    with torch.no_grad():
        upsampled = upsample(sample)
    assert upsampled.shape == (1, 32, 64, 64)
    output_slice = upsampled[0, -1, -3:, -3:]
    expected_slice = torch.tensor([0.7145, 1.3773, 0.3492, 0.8448, 1.0839, 
        -0.3341, 0.5956, 0.125, -0.4841])
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
