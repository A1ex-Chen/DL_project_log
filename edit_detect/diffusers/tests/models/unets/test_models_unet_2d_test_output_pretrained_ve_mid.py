@slow
def test_output_pretrained_ve_mid(self):
    model = UNet2DModel.from_pretrained('google/ncsnpp-celebahq-256')
    model.to(torch_device)
    batch_size = 4
    num_channels = 3
    sizes = 256, 256
    noise = torch.ones((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor(batch_size * [0.0001]).to(torch_device)
    with torch.no_grad():
        output = model(noise, time_step).sample
    output_slice = output[0, -3:, -3:, -1].flatten().cpu()
    expected_output_slice = torch.tensor([-4836.2178, -6487.147, -3816.8196,
        -7964.9302, -10966.3037, -20043.5957, 8137.0513, 2340.3328, 544.6056])
    self.assertTrue(torch_all_close(output_slice, expected_output_slice,
        rtol=0.01))
