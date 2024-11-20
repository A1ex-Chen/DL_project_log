def callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
    callback_fn.has_been_called = True
    nonlocal number_of_steps
    number_of_steps += 1
    if step == 1:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([-0.2463, -0.4644, -0.9756, 1.5176, 
            1.4414, 0.7866, 0.9897, 0.8521, 0.7983])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
    elif step == 2:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([-0.2644, -0.4626, -0.9653, 1.5176, 
            1.4551, 0.7686, 0.9805, 0.8452, 0.8115])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
