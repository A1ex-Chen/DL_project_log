def test_stable_diffusion_img2img_intermediate_state(self):
    number_of_steps = 0

    def callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
        callback_fn.has_been_called = True
        nonlocal number_of_steps
        number_of_steps += 1
        if step == 1:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 64, 96)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([-0.4958, 0.5107, 1.1045, 2.7539, 
                4.668, 3.832, 1.5049, 1.8633, 2.6523])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
        elif step == 2:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 64, 96)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([-0.4956, 0.5078, 1.0918, 2.752, 
                4.6484, 3.8125, 1.5146, 1.8633, 2.6367])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
    callback_fn.has_been_called = False
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', safety_checker=None, torch_dtype=
        torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs(torch_device, dtype=torch.float16)
    pipe(**inputs, callback=callback_fn, callback_steps=1)
    assert callback_fn.has_been_called
    assert number_of_steps == 2
