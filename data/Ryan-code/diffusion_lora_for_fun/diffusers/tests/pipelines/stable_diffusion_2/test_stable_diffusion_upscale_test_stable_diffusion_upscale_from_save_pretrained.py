def test_stable_diffusion_upscale_from_save_pretrained(self):
    pipes = []
    device = 'cpu'
    low_res_scheduler = DDPMScheduler()
    scheduler = DDIMScheduler(prediction_type='v_prediction')
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd_pipe = StableDiffusionUpscalePipeline(unet=self.
        dummy_cond_unet_upscale, low_res_scheduler=low_res_scheduler,
        scheduler=scheduler, vae=self.dummy_vae, text_encoder=self.
        dummy_text_encoder, tokenizer=tokenizer, max_noise_level=350)
    sd_pipe = sd_pipe.to(device)
    pipes.append(sd_pipe)
    with tempfile.TemporaryDirectory() as tmpdirname:
        sd_pipe.save_pretrained(tmpdirname)
        sd_pipe = StableDiffusionUpscalePipeline.from_pretrained(tmpdirname
            ).to(device)
    pipes.append(sd_pipe)
    prompt = 'A painting of a squirrel eating a burger'
    image = self.dummy_image.cpu().permute(0, 2, 3, 1)[0]
    low_res_image = Image.fromarray(np.uint8(image)).convert('RGB').resize((
        64, 64))
    image_slices = []
    for pipe in pipes:
        generator = torch.Generator(device=device).manual_seed(0)
        image = pipe([prompt], image=low_res_image, generator=generator,
            guidance_scale=6.0, noise_level=20, num_inference_steps=2,
            output_type='np').images
        image_slices.append(image[0, -3:, -3:, -1].flatten())
    assert np.abs(image_slices[0] - image_slices[1]).max() < 0.001
