def test_unclip_passed_image_embed(self):
    device = torch.device('cpu')


    class DummyScheduler:
        init_noise_sigma = 1
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=None)
    generator = torch.Generator(device=device).manual_seed(0)
    dtype = pipe.decoder.dtype
    batch_size = 1
    shape = (batch_size, pipe.decoder.config.in_channels, pipe.decoder.
        config.sample_size, pipe.decoder.config.sample_size)
    decoder_latents = pipe.prepare_latents(shape, dtype=dtype, device=
        device, generator=generator, latents=None, scheduler=DummyScheduler())
    shape = (batch_size, pipe.super_res_first.config.in_channels // 2, pipe
        .super_res_first.config.sample_size, pipe.super_res_first.config.
        sample_size)
    super_res_latents = pipe.prepare_latents(shape, dtype=dtype, device=
        device, generator=generator, latents=None, scheduler=DummyScheduler())
    pipeline_inputs = self.get_dummy_inputs(device, pil_image=False)
    img_out_1 = pipe(**pipeline_inputs, decoder_latents=decoder_latents,
        super_res_latents=super_res_latents).images
    pipeline_inputs = self.get_dummy_inputs(device, pil_image=False)
    image = pipeline_inputs.pop('image')
    image_embeddings = pipe.image_encoder(image).image_embeds
    img_out_2 = pipe(**pipeline_inputs, decoder_latents=decoder_latents,
        super_res_latents=super_res_latents, image_embeddings=image_embeddings
        ).images
    assert np.abs(img_out_1 - img_out_2).max() < 0.0001
