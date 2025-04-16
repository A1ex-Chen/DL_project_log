def save_blip_diffusion_model(model, args):
    qformer = get_qformer(model)
    qformer.eval()
    text_encoder = ContextCLIPTextModel.from_pretrained(
        'runwayml/stable-diffusion-v1-5', subfolder='text_encoder')
    vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5',
        subfolder='vae')
    unet = UNet2DConditionModel.from_pretrained(
        'runwayml/stable-diffusion-v1-5', subfolder='unet')
    vae.eval()
    text_encoder.eval()
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', set_alpha_to_one=False,
        skip_prk_steps=True)
    tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5',
        subfolder='tokenizer')
    image_processor = BlipImageProcessor()
    blip_diffusion = BlipDiffusionPipeline(tokenizer=tokenizer,
        text_encoder=text_encoder, vae=vae, unet=unet, scheduler=scheduler,
        qformer=qformer, image_processor=image_processor)
    blip_diffusion.save_pretrained(args.checkpoint_path)
