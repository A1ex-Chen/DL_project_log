def test_stable_diffusion_panorama_intermediate_state(self):
    number_of_steps = 0

    def callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
        callback_fn.has_been_called = True
        nonlocal number_of_steps
        number_of_steps += 1
        if step == 1:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 64, 256)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([0.18681869, 0.33907816, 0.5361276, 
                0.14432865, -0.02856611, -0.73941123, 0.23397987, 
                0.47322682, -0.37823164])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
        elif step == 2:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 64, 256)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([0.18539645, 0.33987248, 0.5378559, 
                0.14437142, -0.02455261, -0.7338317, 0.23990755, 0.47356272,
                -0.3786505])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
    callback_fn.has_been_called = False
    model_ckpt = 'stabilityai/stable-diffusion-2-base'
    scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder='scheduler'
        )
    pipe = StableDiffusionPanoramaPipeline.from_pretrained(model_ckpt,
        scheduler=scheduler, safety_checker=None)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    inputs = self.get_inputs()
    pipe(**inputs, callback=callback_fn, callback_steps=1)
    assert callback_fn.has_been_called
    assert number_of_steps == 3