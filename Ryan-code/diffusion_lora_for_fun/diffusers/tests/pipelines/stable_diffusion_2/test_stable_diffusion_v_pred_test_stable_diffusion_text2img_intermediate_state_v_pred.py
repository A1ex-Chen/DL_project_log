def test_stable_diffusion_text2img_intermediate_state_v_pred(self):
    number_of_steps = 0

    def test_callback_fn(step: int, timestep: int, latents: torch.Tensor
        ) ->None:
        test_callback_fn.has_been_called = True
        nonlocal number_of_steps
        number_of_steps += 1
        if step == 0:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 96, 96)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([0.7749, 0.0325, 0.5088, 0.1619, 
                0.3372, 0.3667, -0.5186, 0.686, 1.4326])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
        elif step == 19:
            latents = latents.detach().cpu().numpy()
            assert latents.shape == (1, 4, 96, 96)
            latents_slice = latents[0, -3:, -3:, -1]
            expected_slice = np.array([1.3887, 1.0273, 1.7266, 0.0726, 
                0.6611, 0.1598, -1.0547, 0.1522, 0.0227])
            assert np.abs(latents_slice.flatten() - expected_slice).max(
                ) < 0.05
    test_callback_fn.has_been_called = False
    pipe = StableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2', torch_dtype=torch.float16)
    pipe = pipe.to(torch_device)
    pipe.set_progress_bar_config(disable=None)
    pipe.enable_attention_slicing()
    prompt = 'Andromeda galaxy in a bottle'
    generator = torch.manual_seed(0)
    pipe(prompt=prompt, num_inference_steps=20, guidance_scale=7.5,
        generator=generator, callback=test_callback_fn, callback_steps=1)
    assert test_callback_fn.has_been_called
    assert number_of_steps == 20
