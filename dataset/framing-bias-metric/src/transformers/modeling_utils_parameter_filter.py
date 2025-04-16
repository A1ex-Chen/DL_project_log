def parameter_filter(x):
    return (x.requires_grad or not only_trainable) and not (isinstance(x,
        torch.nn.Embedding) and exclude_embeddings)
