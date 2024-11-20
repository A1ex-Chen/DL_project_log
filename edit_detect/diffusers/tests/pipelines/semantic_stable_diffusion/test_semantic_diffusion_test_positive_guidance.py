def test_positive_guidance(self):
    torch_device = 'cuda'
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5')
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
    expected_slice = [0.34673113, 0.38492733, 0.37597352, 0.34086335, 
        0.35650748, 0.35579205, 0.3384763, 0.34340236, 0.3573271]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    generator.manual_seed(seed)
    output = pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, **edit)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.41887826, 0.37728766, 0.30138272, 0.41416335, 
        0.41664985, 0.36283392, 0.36191246, 0.43364465, 0.43001732]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
