def test_resnet_up(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64).to(torch_device)
    temb = torch.randn(1, 128).to(torch_device)
    resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, up=True
        ).to(torch_device)
    with torch.no_grad():
        output_tensor = resnet_block(sample, temb)
    assert output_tensor.shape == (1, 32, 128, 128)
    output_slice = output_tensor[0, -1, -3:, -3:]
    expected_slice = torch.tensor([1.213, -0.8753, -0.9027, 1.5783, -0.5362,
        -0.5001, 1.0726, -0.7732, -0.4182], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
