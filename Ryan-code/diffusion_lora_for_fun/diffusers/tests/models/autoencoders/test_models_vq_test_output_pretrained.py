def test_output_pretrained(self):
    model = VQModel.from_pretrained('fusing/vqgan-dummy')
    model.to(torch_device).eval()
    torch.manual_seed(0)
    backend_manual_seed(torch_device, 0)
    image = torch.randn(1, model.config.in_channels, model.config.
        sample_size, model.config.sample_size)
    image = image.to(torch_device)
    with torch.no_grad():
        output = model(image).sample
    output_slice = output[0, -1, -3:, -3:].flatten().cpu()
    expected_output_slice = torch.tensor([-0.0153, -0.4044, -0.188, -0.5161,
        -0.2418, -0.4072, -0.1612, -0.0633, -0.0143])
    self.assertTrue(torch.allclose(output_slice, expected_output_slice,
        atol=0.001))
