def get_dummy_components(self):
    unet = UniDiffuserModel.from_pretrained(
        'hf-internal-testing/unidiffuser-diffusers-test', subfolder='unet')
    scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=
        0.012, beta_schedule='scaled_linear', solver_order=3)
    vae = AutoencoderKL.from_pretrained(
        'hf-internal-testing/unidiffuser-diffusers-test', subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(
        'hf-internal-testing/unidiffuser-diffusers-test', subfolder=
        'text_encoder')
    clip_tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/unidiffuser-diffusers-test', subfolder=
        'clip_tokenizer')
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'hf-internal-testing/unidiffuser-diffusers-test', subfolder=
        'image_encoder')
    clip_image_processor = CLIPImageProcessor(crop_size=32, size=32)
    text_tokenizer = GPT2Tokenizer.from_pretrained(
        'hf-internal-testing/unidiffuser-diffusers-test', subfolder=
        'text_tokenizer')
    text_decoder = UniDiffuserTextDecoder.from_pretrained(
        'hf-internal-testing/unidiffuser-diffusers-test', subfolder=
        'text_decoder')
    components = {'vae': vae, 'text_encoder': text_encoder, 'image_encoder':
        image_encoder, 'clip_image_processor': clip_image_processor,
        'clip_tokenizer': clip_tokenizer, 'text_decoder': text_decoder,
        'text_tokenizer': text_tokenizer, 'unet': unet, 'scheduler': scheduler}
    return components
