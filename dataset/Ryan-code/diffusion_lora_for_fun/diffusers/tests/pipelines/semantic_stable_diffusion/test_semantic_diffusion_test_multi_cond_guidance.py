def test_multi_cond_guidance(self):
    torch_device = 'cuda'
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5')
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    prompt = 'a castle next to a river'
    edit = {'editing_prompt': ['boat on a river, boat',
        'monet, impression, sunrise'], 'reverse_editing_direction': False,
        'edit_warmup_steps': [15, 18], 'edit_guidance_scale': 6,
        'edit_threshold': [0.9, 0.8], 'edit_momentum_scale': 0.5,
        'edit_mom_beta': 0.6}
    seed = 48
    guidance_scale = 7
    generator = torch.Generator(torch_device)
    generator.manual_seed(seed)
    output = pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.75163555, 0.76037145, 0.61785, 0.9189673, 0.8627701,
        0.85189694, 0.8512813, 0.87012076, 0.8312857]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
    generator.manual_seed(seed)
    output = pipe([prompt], generator=generator, guidance_scale=
        guidance_scale, num_inference_steps=50, output_type='np', width=512,
        height=512, **edit)
    image = output.images
    image_slice = image[0, -3:, -3:, -1]
    expected_slice = [0.73553365, 0.7537271, 0.74341905, 0.66480356, 
        0.6472925, 0.63039416, 0.64812905, 0.6749717, 0.6517102]
    assert image.shape == (1, 512, 512, 3)
    assert np.abs(image_slice.flatten() - expected_slice).max() < 0.01
