def test_stable_diffusion_xl_inpaint_mask_latents(self):
    device = 'cpu'
    components = self.get_dummy_components()
    sd_pipe = self.pipeline_class(**components).to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    inputs = self.get_dummy_inputs(device)
    inputs['strength'] = 0.9
    out_0 = sd_pipe(**inputs).images
    inputs = self.get_dummy_inputs(device)
    image = sd_pipe.image_processor.preprocess(inputs['image']).to(sd_pipe.
        device)
    mask = sd_pipe.mask_processor.preprocess(inputs['mask_image']).to(sd_pipe
        .device)
    masked_image = image * (mask < 0.5)
    generator = torch.Generator(device=device).manual_seed(0)
    image_latents = sd_pipe._encode_vae_image(image, generator=generator)
    torch.randn((1, 4, 32, 32), generator=generator)
    mask_latents = sd_pipe._encode_vae_image(masked_image, generator=generator)
    inputs['image'] = image_latents
    inputs['masked_image_latents'] = mask_latents
    inputs['mask_image'] = mask
    inputs['strength'] = 0.9
    generator = torch.Generator(device=device).manual_seed(0)
    torch.randn((1, 4, 32, 32), generator=generator)
    inputs['generator'] = generator
    out_1 = sd_pipe(**inputs).images
    assert np.abs(out_0 - out_1).max() < 0.01
