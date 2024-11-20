def callback_fn(step: int, timestep: int, latents: torch.Tensor) ->None:
    callback_fn.has_been_called = True
    nonlocal number_of_steps
    number_of_steps += 1
    if step == 1:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([-0.5693, -0.3018, -0.9746, 0.0518, -
            0.877, 0.7559, -1.7402, 0.1022, 1.1582])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
    elif step == 2:
        latents = latents.detach().cpu().numpy()
        assert latents.shape == (1, 4, 64, 64)
        latents_slice = latents[0, -3:, -3:, -1]
        expected_slice = np.array([-0.1958, -0.2993, -1.0166, -0.5005, -
            0.481, 0.6162, -0.9492, 0.6621, 1.4492])
        assert np.abs(latents_slice.flatten() - expected_slice).max() < 0.05
