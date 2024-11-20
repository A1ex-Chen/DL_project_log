def test_output_pretrained(self):
    model = UNet2DModel.from_pretrained('fusing/unet-ldm-dummy-update')
    model.eval()
    model.to(torch_device)
    noise = torch.randn(1, model.config.in_channels, model.config.
        sample_size, model.config.sample_size, generator=torch.manual_seed(0))
    noise = noise.to(torch_device)
    time_step = torch.tensor([10] * noise.shape[0]).to(torch_device)
    with torch.no_grad():
        output = model(noise, time_step).sample
    output_slice = output[0, -1, -3:, -3:].flatten().cpu()
    expected_output_slice = torch.tensor([-13.3258, -20.11, -15.9873, -
        17.6617, -23.0596, -17.9419, -13.3675, -16.1889, -12.38])
    self.assertTrue(torch_all_close(output_slice, expected_output_slice,
        rtol=0.001))
