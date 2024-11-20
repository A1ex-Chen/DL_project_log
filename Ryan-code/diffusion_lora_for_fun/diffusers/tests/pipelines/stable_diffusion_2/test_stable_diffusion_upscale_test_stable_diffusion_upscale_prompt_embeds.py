def test_stable_diffusion_upscale_prompt_embeds(self):
    device = 'cpu'
    unet = self.dummy_cond_unet_upscale
    low_res_scheduler = DDPMScheduler()
    scheduler = DDIMScheduler(prediction_type='v_prediction')
    vae = self.dummy_vae
    text_encoder = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
    low_res_image = Image.fromarray(np.uint8(image)).convert('RGB').resize((
        64, 64))
    sd_pipe = StableDiffusionUpscalePipeline(unet=unet, low_res_scheduler=
        low_res_scheduler, scheduler=scheduler, vae=vae, text_encoder=
        text_encoder, tokenizer=tokenizer, max_noise_level=350)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.Generator(device=device).manual_seed(0)
    output = sd_pipe([prompt], image=low_res_image, generator=generator,
        guidance_scale=6.0, noise_level=20, num_inference_steps=2,
        output_type='np')
    image = output.images
    generator = torch.Generator(device=device).manual_seed(0)
    prompt_embeds, negative_prompt_embeds = sd_pipe.encode_prompt(prompt,
        device, 1, False)
    if negative_prompt_embeds is not None:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    image_from_prompt_embeds = sd_pipe(prompt_embeds=prompt_embeds, image=[
        low_res_image], generator=generator, guidance_scale=6.0,
        noise_level=20, num_inference_steps=2, output_type='np',
        return_dict=False)[0]
    image_slice = image[0, -3:, -3:, -1]
    image_from_prompt_embeds_slice = image_from_prompt_embeds[0, -3:, -3:, -1]
    expected_height_width = low_res_image.size[0] * 4
    assert image.shape == (1, expected_height_width, expected_height_width, 3)
    expected_slice = np.array([0.3113, 0.391, 0.4272, 0.4859, 0.5061, 
        0.4652, 0.5362, 0.5715, 0.5661])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_prompt_embeds_slice.flatten() - expected_slice
        ).max() < 0.01
