def test_stable_diffusion_img_variation_intermediate_state(self):
    number_of_steps = 0

    def callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
        callback_fn.has_been_called = True
        nonlocal number_of_steps
        number_of_steps += 1
        if step == 1:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 64, 64)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([-0.7974, -0.4343, -1.087, 0.04785, -
                1.327, 0.855, -2.148, -0.1725, 1.439])
            max_diff = numpy_cosine_similarity_distance(latents_slice.
                flatten(), expected_slice)
            assert max_diff < 0.001
        elif step == 2:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 64, 64)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([0.3232, 0.004883, 0.913, -1.084, 
                0.6143, -1.6875, -2.463, -0.439, -0.419])
            max_diff = numpy_cosine_similarity_distance(latents_slice.
                flatten(), expected_slice)
            assert max_diff < 0.001
    callback_fn.has_been_called = False
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        'lambdalabs/sd-image-variations-diffusers', safety_checker=None,
        torch_dtype=torch.float16)
    pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    for component in pipe.components.values():
        if hasattr(component, 'set_default_attn_processor'):
            component.set_default_attn_processor()
    generator_device = 'cpu'
    inputs = self.get_inputs(generator_device, dtype=torch.float16)
    pipe(**inputs, callback=callback_fn, callback_steps=1)
    assert callback_fn.has_been_called
    assert number_of_steps == inputs['num_inference_steps']
