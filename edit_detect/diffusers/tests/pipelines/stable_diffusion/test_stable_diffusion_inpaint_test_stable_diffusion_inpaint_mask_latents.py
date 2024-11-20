def test_stable_diffusion_inpaint_mask_latents(self):
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
    image_latents = sd_pipe.vae.encode(image).latent_dist.sample(generator=
        generator) * sd_pipe.vae.config.scaling_factor
    torch.randn((1, 4, 32, 32), generator=generator)
    mask_latents = sd_pipe.vae.encode(masked_image).latent_dist.sample(
        generator=generator) * sd_pipe.vae.config.scaling_factor
    inputs['image'] = image_latents
    inputs['masked_image_latents'] = mask_latents
    inputs['mask_image'] = mask
    inputs['strength'] = 0.9
    generator = torch.Generator(device=device).manual_seed(0)
    torch.randn((1, 4, 32, 32), generator=generator)
    inputs['generator'] = generator
    out_1 = sd_pipe(**inputs).images
    assert np.abs(out_0 - out_1).max() < 0.01
