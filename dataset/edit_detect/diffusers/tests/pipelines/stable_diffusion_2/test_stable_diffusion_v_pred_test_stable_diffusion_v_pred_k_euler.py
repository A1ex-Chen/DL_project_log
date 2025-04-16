def test_stable_diffusion_v_pred_k_euler(self):
    device = 'cpu'
    unet = self.dummy_cond_unet
    scheduler = EulerDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', prediction_type='v_prediction')
    vae = self.dummy_vae
    bert = self.dummy_text_encoder
    tokenizer = CLIPTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-clip')
    sd_pipe = StableDiffusionPipeline(unet=unet, scheduler=scheduler, vae=
        vae, text_encoder=bert, tokenizer=tokenizer, safety_checker=None,
        feature_extractor=None, image_encoder=None, requires_safety_checker
        =False)
    sd_pipe = sd_pipe.to(device)
    sd_pipe.set_progress_bar_config(disable=None)
    prompt = 'A painting of a squirrel eating a burger'
    generator = torch.Generator(device=device).manual_seed(0)
    output = sd_pipe([prompt], generator=generator, guidance_scale=6.0,
        num_inference_steps=2, output_type='np')
    image = output.images
    generator = torch.Generator(device=device).manual_seed(0)
    image_from_tuple = sd_pipe([prompt], generator=generator,
        guidance_scale=6.0, num_inference_steps=2, output_type='np',
        return_dict=False)[0]
    image_slice = image[0, -3:, -3:, -1]
    image_from_tuple_slice = image_from_tuple[0, -3:, -3:, -1]
    assert image.shape == (1, 64, 64, 3)
    expected_slice = np.array([0.5644, 0.6514, 0.519, 0.5663, 0.5287, 
        0.4953, 0.543, 0.5243, 0.4778])
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    assert np.abs(image_from_tuple_slice.flatten() - expected_slice).max(
        ) < 0.01
