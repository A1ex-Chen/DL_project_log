def test_inference(self):
    unet = self.dummy_uncond_unet
    scheduler = PNDMScheduler()
    pndm = PNDMPipeline(unet=unet, scheduler=scheduler)
    pndm.to(torch_device)
    pndm.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    image = pndm(generator=generator, num_inference_steps=20, output_type='np'
        ).images
    generator = torch.manual_seed(0)
    image_from_tuple = pndm(generator=generator, num_inference_steps=20,
        output_type='np', return_dict=False)[0]
    image_slice = image[0, -3:, -3:, -1]
    image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
    assert image.shape == (1, 32, 32, 3)
    expected_slice = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01
