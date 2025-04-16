def test_upsample_with_transpose(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 32, 32)
    upsample = Upsample2D(channels=32, use_conv=False, use_conv_transpose=True)
    with torch.no_grad():
        upsampled = upsample(sample)
    assert upsampled.shape == (1, 32, 64, 64)
    output_slice = upsampled[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-0.3028, -0.1582, 0.0071, 0.035, -0.4799,
        -0.1139, 0.1056, -0.1153, -0.1046])
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
