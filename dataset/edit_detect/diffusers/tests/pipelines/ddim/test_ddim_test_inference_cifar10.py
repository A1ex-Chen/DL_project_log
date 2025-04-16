def test_inference_cifar10(self):
    model_id = 'google/ddpm-cifar10-32'
    unet = UNet2DModel.from_pretrained(model_id)
    scheduler = DDIMScheduler()
    ddim = DDIMPipeline(unet=unet, scheduler=scheduler)
    ddim.to(torch_device)
    ddim.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    image = ddim(generator=generator, eta=0.0, output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.1723, 0.1617, 0.16, 0.1626, 0.1497, 0.1513,
        0.1505, 0.1442, 0.1453])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
