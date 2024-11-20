def test_upsample_default(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 32, 32)
    upsample = Upsample2D(channels=32, use_conv=False)
    with torch.no_grad():
        upsampled = upsample(sample)
    assert upsampled.shape == (1, 32, 64, 64)
    output_slice = upsampled[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-0.2173, -1.2079, -1.2079, 0.2952, 
        1.1254, 1.1254, 0.2952, 1.1254, 1.1254])
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
