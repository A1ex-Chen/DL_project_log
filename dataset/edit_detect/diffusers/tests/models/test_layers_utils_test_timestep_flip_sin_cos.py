def test_timestep_flip_sin_cos(self):
    embedding_dim = 16
    timesteps = torch.arange(10)
    t1 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)
    t1 = torch.cat([t1[:, embedding_dim // 2:], t1[:, :embedding_dim // 2]],
        dim=-1)
    t2 = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False
        )
    assert torch.allclose(t1.cpu(), t2.cpu(), 0.001)
