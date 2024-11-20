def test_callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
    test_callback_fn.has_been_called = True
    nonlocal number_of_steps
    number_of_steps += 1
    if step == 0:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 96, 96)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([0.7749, 0.0325, 0.5088, 0.1619, 0.3372, 
            0.3667, -0.5186, 0.686, 1.4326])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
    elif step == 19:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 96, 96)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([1.3887, 1.0273, 1.7266, 0.0726, 0.6611, 
            0.1598, -1.0547, 0.1522, 0.0227])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
