def test_inference_cifar10(self):
    model_id = 'google/ddpm-cifar10-32'
    unet = UNet2DModel.from_pretrained(model_id)
    scheduler = DDPMScheduler.from_pretrained(model_id)
    ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
    ddpm.to(torch_device)
    ddpm.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    image = ddpm(generator=generator, output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.42, 0.3588, 0.1939, 0.3847, 0.3382, 0.2647,
        0.4155, 0.3582, 0.3385])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
