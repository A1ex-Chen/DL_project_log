def test_guidance_fp16(self):
    torch_device = 'cuda'
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'a photo of a cat'
    edit = {'editing_prompt': ['sunglasses'], 'reverse_editing_direction':
        [False], 'edit_warmup_steps': 10, 'edit_guidance_scale': 6,
        'edit_threshold': 0.95, 'edit_momentum_scale': 0.5, 'edit_mom_beta':
        0.6}
    seed = 3
    guidance_scale = 7
    generator = torch.Generator(torch_device)
    generator.manual_seed(seed)
    output = pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.34887695, 0.3876953, 0.375, 0.34423828, 0.3581543, 
        0.35717773, 0.3383789, 0.34570312, 0.359375]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    generator.manual_seed(seed)
    output = pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, **edit)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.42285156, 0.36914062, 0.29077148, 0.42041016, 
        0.41918945, 0.35498047, 0.3618164, 0.4423828, 0.43115234]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
