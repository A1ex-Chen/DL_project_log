def test_sinoid_embeddings_hardcoded(self):
    embedding_dim = 64
    timesteps = torch.arange(128)
    t1 = get_timestep_embedding(timesteps, embedding_dim,
        downscale_freq_shift=1, flip_sin_to_cos=False)
    t2 = get_timestep_embedding(timesteps, embedding_dim,
        downscale_freq_shift=0, flip_sin_to_cos=True)
    t3 = get_timestep_embedding(timesteps, embedding_dim, scale=1000)
    assert torch.allclose(t1[23:26, 47:50].flatten().cpu(), torch.tensor([
        0.9646, 0.9804, 0.9892, 0.9615, 0.9787, 0.9882, 0.9582, 0.9769, 
        0.9872]), 0.001)
    assert torch.allclose(t2[23:26, 47:50].flatten().cpu(), torch.tensor([
        0.3019, 0.228, 0.1716, 0.3146, 0.2377, 0.179, 0.3272, 0.2474, 
        0.1864]), 0.001)
    assert torch.allclose(t3[23:26, 47:50].flatten().cpu(), torch.tensor([-
        0.9801, -0.9464, -0.9349, -0.3952, 0.8887, -0.9709, 0.5299, -0.2853,
        -0.9927]), 0.001)
