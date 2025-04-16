def test_fast_inference(self):
    device = 'cpu'
    unet = self.dummy_uncond_unet
    scheduler = DDPMScheduler()
    ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
    ddpm.to(device)
    ddpm.set_progress_bar_config(disable=None)
    generator = torch.Generator(device=device).manual_seed(0)
    image = ddpm(generator=generator, num_inference_steps=2, output_type='np'
        ).images
    generator = torch.Generator(device=device).manual_seed(0)
    image_from_tuple = ddpm(generator=generator, num_inference_steps=2,
        output_type='np', return_dict=False)[0]
    image_slice = image[0, -3:, -3:, -1]
    image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
    assert image.shape == (1, 8, 8, 3)
    expected_slice = np.array([0.0, 0.9996672, 0.00329116, 1.0, 0.9995991, 
        1.0, 0.0060907, 0.00115037, 0.0])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01
