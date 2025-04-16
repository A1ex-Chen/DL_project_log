@unittest.skipIf(torch_device != 'cuda', 'This test requires a GPU')
def test_stable_diffusion_upscale_fp16(self):
    """Test that stable diffusion upscale works with fp16"""
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
    unet = unet.half()
    text_encoder = text_encoder.half()
    sd_pipe = StableDiffusionUpscalePipeline(unet=unet, low_res_scheduler=
        low_res_scheduler, scheduler=scheduler, vae=vae, text_encoder=
        text_encoder, tokenizer=tokenizer, max_noise_level=350)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    image = sd_pipe([prompt], image=low_res_image, generator=generator,
        num_inference_steps=2, output_type='np').images
    expected_height_width = low_res_image.size[0] * 4
    assert image.shape == (1, expected_height_width, expected_height_width, 3)
