def test_ledits_pp_editing(self):
    pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
        'runwayml/stable-diffusion-v1-5', safety_checker=None, torch_dtype=
        torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    generator = torch.manual_seed(0)
    _ = pipe.invert(image=self.raw_image, generator=generator)
    generator = torch.manual_seed(0)
    inputs = {'generator': generator, 'editing_prompt': ['cat', 'dog'],
        'reverse_editing_direction': [True, False], 'edit_guidance_scale':
        [5.0, 5.0], 'edit_threshold': [0.8, 0.8]}
    reconstruction = pipe(**inputs, output_type='np').images[0]
    output_slice = reconstruction[150:153, 140:143, -1]
    output_slice = output_slice.flatten()
    expected_slice = np.array([0.9453125, 0.93310547, 0.84521484, 
        0.94628906, 0.9111328, 0.80859375, 0.93847656, 0.9042969, 0.8144531])
    assert np.abs(output_slice - expected_slice).max() < 0.01
