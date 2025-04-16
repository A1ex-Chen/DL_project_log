def test_unclip_passed_text_embed(self):
    device = torch.device('cpu')


    class DummyScheduler:
        init_noise_sigma = 1
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe = pipe.to(device)
    prior = components['prior']
    decoder = components['decoder']
    super_res_first = components['super_res_first']
    tokenizer = components['tokenizer']
    text_encoder = components['text_encoder']
    generator = torch.Generator(device=device).manual_seed(0)
    dtype = prior.dtype
    batch_size = 1
    shape = batch_size, prior.config.embedding_dim
    prior_latents = pipe.prepare_latents(shape, dtype=dtype, device=device,
        generator=generator, latents=None, scheduler=DummyScheduler())
    shape = (batch_size, decoder.config.in_channels, decoder.config.
        sample_size, decoder.config.sample_size)
    decoder_latents = pipe.prepare_latents(shape, dtype=dtype, device=
        device, generator=generator, latents=None, scheduler=DummyScheduler())
    shape = (batch_size, super_res_first.config.in_channels // 2,
        super_res_first.config.sample_size, super_res_first.config.sample_size)
    super_res_latents = pipe.prepare_latents(shape, dtype=dtype, device=
        device, generator=generator, latents=None, scheduler=DummyScheduler())
    pipe.set_progress_bar_config(disable=None)
    prompt = 'this is a prompt example'
    generator = torch.Generator(device=device).manual_seed(0)
    output = pipe([prompt], generator=generator, prior_num_inference_steps=
        2, decoder_num_inference_steps=2, super_res_num_inference_steps=2,
        prior_latents=prior_latents, decoder_latents=decoder_latents,
        super_res_latents=super_res_latents, output_type='np')
    image = output.images
    text_inputs = tokenizer(prompt, padding='max_length', max_length=
        tokenizer.model_max_length, return_tensors='pt')
    text_model_output = text_encoder(text_inputs.input_ids)
    text_attention_mask = text_inputs.attention_mask
    generator = torch.Generator(device=device).manual_seed(0)
    image_from_text = pipe(generator=generator, prior_num_inference_steps=2,
        decoder_num_inference_steps=2, super_res_num_inference_steps=2,
        prior_latents=prior_latents, decoder_latents=decoder_latents,
        super_res_latents=super_res_latents, text_model_output=
        text_model_output, text_attention_mask=text_attention_mask,
        output_type='np')[0]
    assert np.abs(image - image_from_text).max() < 0.0001
