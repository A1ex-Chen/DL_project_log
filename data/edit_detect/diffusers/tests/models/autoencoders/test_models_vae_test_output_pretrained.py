def test_output_pretrained(self):
    model = AutoencoderKL.from_pretrained('fusing/autoencoder-kl-dummy')
    model = model.to(torch_device)
    model.eval()
    generator_device = 'cpu' if not torch_device.startswith('cuda') else 'cuda'
    if torch_device != 'mps':
        generator = torch.Generator(device=generator_device).manual_seed(0)
    else:
        generator = torch.manual_seed(0)
    image = torch.randn(1, model.config.in_channels, model.config.
        sample_size, model.config.sample_size, generator=torch.manual_seed(0))
    image = image.to(torch_device)
    with torch.no_grad():
        output = model(image, sample_posterior=True, generator=generator
            ).sample
    output_slice = output[0, -1, -3:, -3:].flatten().cpu()
    if torch_device == 'mps':
        expected_output_slice = torch.tensor([-0.40078, -0.00038323, -
            0.12681, -0.11462, 0.20095, 0.10893, -0.088247, -0.30361, -
            0.0098644])
    elif generator_device == 'cpu':
        expected_output_slice = torch.tensor([-0.1352, 0.0878, 0.0419, -
            0.0818, -0.1069, 0.0688, -0.1458, -0.4446, -0.0026])
    else:
        expected_output_slice = torch.tensor([-0.2421, 0.4642, 0.2507, -
            0.0438, 0.0682, 0.316, -0.2018, -0.0727, 0.2485])
    self.assertTrue(torch_all_close(output_slice, expected_output_slice,
        rtol=0.01))
