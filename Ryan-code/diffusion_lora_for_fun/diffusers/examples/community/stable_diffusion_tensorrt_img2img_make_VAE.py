def make_VAE(model, device, max_batch_size, embedding_dim, inpaint=False):
    return VAE(model, device=device, max_batch_size=max_batch_size,
        embedding_dim=embedding_dim)
