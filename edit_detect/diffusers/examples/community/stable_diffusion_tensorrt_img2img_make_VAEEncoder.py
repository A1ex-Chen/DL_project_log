def make_VAEEncoder(model, device, max_batch_size, embedding_dim, inpaint=False
    ):
    return VAEEncoder(model, device=device, max_batch_size=max_batch_size,
        embedding_dim=embedding_dim)
