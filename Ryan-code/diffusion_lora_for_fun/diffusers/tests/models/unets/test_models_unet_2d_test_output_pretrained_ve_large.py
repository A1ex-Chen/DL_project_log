def test_output_pretrained_ve_large(self):
    model = UNet2DModel.from_pretrained('fusing/ncsnpp-ffhq-ve-dummy-update')
    model.to(torch_device)
    batch_size = 4
    num_channels = 3
    sizes = 32, 32
    noise = torch.ones((batch_size, num_channels) + sizes).to(torch_device)
    time_step = torch.tensor(batch_size * [0.0001]).to(torch_device)
    with torch.no_grad():
        output = model(noise, time_step).sample
    output_slice = output[0, -3:, -3:, -1].flatten().cpu()
    expected_output_slice = torch.tensor([-0.0325, -0.09, -0.0869, -0.0332,
        -0.0725, -0.027, -0.0101, 0.0227, 0.0256])
    self.assertTrue(torch_all_close(output_slice, expected_output_slice,
        rtol=0.01))
