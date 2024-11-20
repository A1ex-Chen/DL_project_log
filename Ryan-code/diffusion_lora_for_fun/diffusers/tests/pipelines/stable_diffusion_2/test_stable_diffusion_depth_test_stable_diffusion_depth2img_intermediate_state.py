def test_stable_diffusion_depth2img_intermediate_state(self):
    number_of_steps = 0

    def callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
        callback_fn.has_been_called = True
        nonlocal number_of_steps
        number_of_steps += 1
        if step == 1:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 60, 80)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([-0.7168, -1.5137, -0.1418, -2.9219, 
                -2.7266, -2.4414, -2.1035, -3.0078, -1.7051])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
        elif step == 2:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 60, 80)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([-0.7109, -1.5068, -0.1403, -2.916, -
                2.7207, -2.4414, -2.1035, -3.0059, -1.709])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
    callback_fn.has_been_called = False
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-depth', safety_checker=None,
        torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(dtype=torch.float16)
    pipe(**inputs, callback=callback_fn, callback_steps=1)
    assert callback_fn.has_been_called
    assert number_of_steps == 2
