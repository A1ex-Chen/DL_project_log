def test_stable_diffusion_components(self):
    """Test that components property works correctly"""
    unet = self.dummy_cond_unet()
    scheduler = PNDMScheduler(skip_prk_steps=True)
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    image = self.dummy_image().cpu().permute(0, 2, 3, 1)[0]
    init_image = Image.fromarray(np.uint8(image)).convert('RGB')
    mask_image = Image.fromarray(np.uint8(image + 4)).convert('RGB').resize((
        32, 32))
    inpaint = StableDiffusionInpaintPipelineLegacy(unet=unet, scheduler=
        scheduler, vae=vae, text_encoder=bert, tokenizer=tokenizer,
        safety_checker=None, feature_extractor=self.dummy_extractor).to(
        torch_device)
    img2img = StableDiffusionImg2ImgPipeline(**inpaint.components,
        image_encoder=None).to(torch_device)
    text2img = StableDiffusionPipeline(**inpaint.components, image_encoder=None
        ).to(torch_device)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    image_inpaint = inpaint([prompt], generator=generator,
        num_inference_steps=2, output_type='np', image=init_image,
        mask_image=mask_image).images
    image_img2img = img2img([prompt], generator=generator,
        num_inference_steps=2, output_type='np', image=init_image).images
    image_text2img = text2img([prompt], generator=generator,
        num_inference_steps=2, output_type='np').images
    assert image_inpaint.shape == (1, 32, 32, 3)
    assert image_img2img.shape == (1, 32, 32, 3)
    assert image_text2img.shape == (1, 64, 64, 3)
