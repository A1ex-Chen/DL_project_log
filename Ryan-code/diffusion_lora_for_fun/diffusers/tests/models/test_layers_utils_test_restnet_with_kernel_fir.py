def test_restnet_with_kernel_fir(self):
    torch.manual_seed(0)
    sample = torch.randn(1, 32, 64, 64).to(torch_device)
    temb = torch.randn(1, 128).to(torch_device)
    resnet_block = ResnetBlock2D(in_channels=32, temb_channels=128, kernel=
        'fir', down=True).to(torch_device)
    with torch.no_grad():
        output_tensor = resnet_block(sample, temb)
    assert output_tensor.shape == (1, 32, 32, 32)
    output_slice = output_tensor[0, -1, -3:, -3:]
    expected_slice = torch.tensor([-0.0934, -0.5729, 0.0909, -0.271, -
        0.5044, 0.0243, -0.0665, -0.5267, -0.3136], device=torch_device)
    assert torch.allclose(output_slice.flatten(), expected_slice, atol=0.001)
