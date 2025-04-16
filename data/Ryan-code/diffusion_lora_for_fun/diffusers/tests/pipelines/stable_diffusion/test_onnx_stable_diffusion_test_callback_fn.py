def test_callback_fn(step: int, timestep: int, latents: np.ndarray) ->None:
    test_callback_fn.has_been_called = True
    nonlocal number_of_steps
    number_of_steps += 1
    if step == 0:
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([-0.6772, -0.3835, -1.2456, 0.1905, -
            1.0974, 0.6967, -1.9353, 0.0178, 1.0167])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.001
    elif step == 5:
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([-0.3351, 0.2241, -0.1837, -0.2325, -
            0.6577, 0.3393, -0.0241, 0.5899, 1.3875])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.001
