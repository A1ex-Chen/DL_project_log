def test_restnet_with_use_in_shortcut(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64).to(torch_device)
    temb = torch.randn(1, 128).to(torch_device)
    resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128,
        use_in_shortcut=True).to(torch_device)
    with torch.no_grad():
        output_tensor = resnet_block(sample, temb)
    assert output_tensor.shape == (1, 32, 64, 64)
    output_slice = output_tensor[0, -1, -3:, -3:]
    expected_slice = torch.tensor([0.2226, -1.0791, -0.1629, 0.3659, -
        0.2889, -1.2376, 0.0582, 0.9206, 0.0044], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
