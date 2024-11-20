def test_ledits_pp_edit(self):
    pipe = LEditsPPPipelineStableDiffusionXL.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', safety_checker=None,
        add_watermarker=None)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    _ = pipe.invert(image=self.raw_image, generator=generator,
        num_zero_noise_steps=0)
    inputs = {'generator': generator, 'editing_prompt': ['cat', 'dog'],
        'reverse_editing_direction': [True, False], 'edit_guidance_scale':
        [2.0, 4.0], 'edit_threshold': [0.8, 0.8]}
    reconstruction = pipe(**inputs, output_type='np').images[0]
    output_slice = reconstruction[150:153, 140:143, -1]
    output_slice = output_slice.flatten()
    expected_slice = np.array([0.56419, 0.44121838, 0.2765603, 0.5708484, 
        0.42763475, 0.30945742, 0.5387106, 0.4735807, 0.3547244])
    assert np.abs(output_slice - expected_slice).max() < 0.001
