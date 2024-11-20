def test_negative_guidance(self):
    torch_device = 'cuda'
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5')
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'an image of a crowded boulevard, realistic, 4k'
    edit = {'editing_prompt': 'crowd, crowded, people',
        'reverse_editing_direction': True, 'edit_warmup_steps': 10,
        'edit_guidance_scale': 8.3, 'edit_threshold': 0.9,
        'edit_momentum_scale': 0.5, 'edit_mom_beta': 0.6}
    seed = 9
    guidance_scale = 7
    generator = torch.Generator(torch_device)
    generator.manual_seed(seed)
    output = pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.43497998, 0.91814065, 0.7540739, 0.55580205, 
        0.8467265, 0.5389691, 0.62574506, 0.58897763, 0.50926757]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    generator.manual_seed(seed)
    output = pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, **edit)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.3089719, 0.30500144, 0.29016042, 0.30630964, 
        0.325687, 0.29419225, 0.2908091, 0.28723598, 0.27696294]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
