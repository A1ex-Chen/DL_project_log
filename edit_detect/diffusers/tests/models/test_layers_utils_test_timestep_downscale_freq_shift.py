def test_timestep_downscale_freq_shift(self):
    embedding_dim = 16
    timesteps = torch.arange(10)
    t1 = get_timestep_embedding(timesteps, embedding_dim,
        downscale_freq_shift=0)
    t2 = get_timestep_embedding(timesteps, embedding_dim,
        downscale_freq_shift=1)
    cosine_half = (t1 - t2)[:, embedding_dim // 2:]
    assert (np.abs((cosine_half <= 0).numpy()) - 1).sum() < 1e-05
