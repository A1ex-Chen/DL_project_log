def test_resnet_down(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64).to(torch_device)
    temb = torch.randn(1, 128).to(torch_device)
    resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, down=True
        ).to(torch_device)
    with torch.no_grad():
        output_tensor = resnet_block(sample, temb)
    assert output_tensor.shape == (1, 32, 32, 32)
    output_slice = output_tensor[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-0.3002, -0.7135, 0.1359, 0.0561, -
        0.7935, 0.0113, -0.1766, -0.6714, -0.0436], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
