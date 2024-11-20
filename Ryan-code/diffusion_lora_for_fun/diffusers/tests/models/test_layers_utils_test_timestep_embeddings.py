def test_timestep_embeddings(self):
    embedding_dim = 256
    timesteps = torch.arange(16)
    t1 = get_timestep_embedding(timesteps, embedding_dim)
    assert (t1[0, :embedding_dim // 2] - 0).abs().sum() < 1e-05
    assert (t1[0, embedding_dim // 2:] - 1).abs().sum() < 1e-05
    assert (t1[:, -1] - 1).abs().sum() < 1e-05
    grad_mean = np.abs(np.gradient(t1, axis=-1)).mean(axis=1)
    prev_grad = 0.0
    for grad in grad_mean:
        assert grad > prev_grad
        prev_grad = grad
