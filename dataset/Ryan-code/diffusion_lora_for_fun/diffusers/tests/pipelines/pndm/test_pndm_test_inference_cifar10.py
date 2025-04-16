def test_inference_cifar10(self):
    model_id = 'google/ddpm-cifar10-32'
    unet = UNet2DModel.from_pretrained(model_id)
    scheduler = PNDMScheduler()
    pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
    pndm.to(torch_device)
    pndm.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    image = pndm(generator=generator, output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([0.1564, 0.14645, 0.1406, 0.14715, 0.12425, 
        0.14045, 0.13115, 0.12175, 0.125])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
