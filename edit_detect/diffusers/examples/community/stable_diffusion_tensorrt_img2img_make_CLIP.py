def make_CLIP(model, device, max_batch_size, embedding_dim, inpaint=False):
    return CLIP(model, device=device, max_batch_size=max_batch_size,
        embedding_dim=embedding_dim)
