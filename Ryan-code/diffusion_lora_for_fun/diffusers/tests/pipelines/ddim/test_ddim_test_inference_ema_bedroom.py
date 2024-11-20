def test_inference_ema_bedroom(self):
    model_id = 'google/ddpm-ema-bedroom-256'
    unet = UNet2DModel.from_pretrained(model_id)
    scheduler = DDIMScheduler.from_pretrained(model_id)
    ddpm = DDIMPipeline(unet=unet, scheduler=scheduler)
    ddpm.to(torch_device)
    ddpm.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    image = ddpm(generator=generator, output_type='np').images
    image_slice = image[0, -3:, -3:, -1]
    assert image.shape == (1, 256, 256, 3)
    expected_slice = np.array([0.006, 0.0201, 0.0344, 0.0024, 0.0018, 
        0.0002, 0.0022, 0.0, 0.0069])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
