def test_timestep_defaults(self):
    embedding_dim = 16
    timesteps = torch.arange(10)
    t1 = get_timestep_embedding(timesteps, embedding_dim)
    t2 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=
        False, downscale_freq_shift=1, max_period=10000)
    assert torch.allclose(t1.cpu(), t2.cpu(), 0.001)
