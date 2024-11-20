def test_inference_predict_sample(self):
    unet = self.dummy_uncond_unet
    scheduler = DDPMScheduler(prediction_type='sample')
    ddpm = DDPMPipeline(unet=unet, scheduler=scheduler)
    ddpm.to(torch_device)
    ddpm.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    image = ddpm(generator=generator, num_inference_steps=2, output_type='np'
        ).images
    generator = torch.manual_seed(0)
    image_eps = ddpm(generator=generator, num_inference_steps=2,
        output_type='np')[0]
    image_slice = image[0, -3:, -3:, -1]
    image_eps_slice = image_eps[0, -3:, -3:, -1]
    assert image.shape == (1, 8, 8, 3)
    tolerance = 0.01 if torch_device != 'mps' else 0.03
    assert np.abs(image_slice.flatten() - image_eps_slice.flatten()).max(
        ) < tolerance
