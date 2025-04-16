def callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
    callback_fn.has_been_called = True
    nonlocal number_of_steps
    number_of_steps += 1
    if step == 1:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([-0.3862, -0.4507, -1.1729, 0.0686, -
            1.1045, 0.7124, -1.8301, 0.1903, 1.2773])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
    elif step == 2:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([0.272, -0.1863, -0.7383, -0.5029, -
            0.7534, 0.397, -0.7646, 0.4468, 1.2686])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
