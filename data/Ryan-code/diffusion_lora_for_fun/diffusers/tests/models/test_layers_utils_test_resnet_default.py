def test_resnet_default(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64).to(torch_device)
    temb = torch.randn(1, 128).to(torch_device)
    resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128).to(
        torch_device)
    with torch.no_grad():
        output_tensor = resnet_block(sample, temb)
    assert output_tensor.shape == (1, 32, 64, 64)
    output_slice = output_tensor[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-1.901, -0.2974, -0.8245, -1.3533, 
        0.8742, -0.9645, -2.0584, 1.3387, -0.4746], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
