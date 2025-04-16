def test_stable_diffusion_upscale_batch(self):
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
    output = sd_pipe(2 * [prompt], image=2 * [low_res_image],
        guidance_scale=6.0, noise_level=20, num_inference_steps=2,
        output_type='np')
    image = output.images
    assert image.shape[0] == 2
    generator = torch.Generator(device=device).manual_seed(0)
    output = sd_pipe([prompt], image=low_res_image, generator=generator,
        num_images_per_prompt=2, guidance_scale=6.0, noise_level=20,
        num_inference_steps=2, output_type='np')
    image = output.images
    assert image.shape[0] == 2
