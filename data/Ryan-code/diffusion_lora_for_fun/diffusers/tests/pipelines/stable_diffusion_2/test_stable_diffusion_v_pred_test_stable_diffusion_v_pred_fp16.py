@unittest.skipIf(torch_device != 'cuda', 'This test requires a GPU')
def test_stable_diffusion_v_pred_fp16(self):
    """Test that stable diffusion v-prediction works with fp16"""
    unet = self.dummy_cond_unet
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', clip_sample=False, set_alpha_to_one=
        False, prediction_type='v_prediction')
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    unet = unet.half()
    vae = vae.half()
    bert = bert.half()
    sd_pipe = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=
        vae, text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=None, image_encoder=None, requires_safety_checker
        =False)
    sd_pipe = sd_pipe.to(torch_device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.manual_seed(0)
    image = sd_pipe([prompt], generator=generator, num_inference_steps=2,
        output_type='np').images
    assert image.shape == (1, 64, 64, 3)
