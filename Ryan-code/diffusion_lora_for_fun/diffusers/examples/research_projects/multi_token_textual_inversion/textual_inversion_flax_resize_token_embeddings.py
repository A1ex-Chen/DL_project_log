def resize_token_embeddings(model, new_num_tokens, initializer_token_id,
    placeholder_token_id, rng):
    if model.config.vocab_size == new_num_tokens or new_num_tokens is None:
        return
    model.config.vocab_size = new_num_tokens
    params = model.params
    old_embeddings = params['text_model']['embeddings']['token_embedding'][
        'embedding']
    old_num_tokens, emb_dim = old_embeddings.shape
    initializer = jax.nn.initializers.normal()
    new_embeddings = initializer(rng, (new_num_tokens, emb_dim))
    new_embeddings = new_embeddings.at[:old_num_tokens].set(old_embeddings)
    new_embeddings = new_embeddings.at[placeholder_token_id].set(new_embeddings
        [initializer_token_id])
    params['text_model']['embeddings']['token_embedding']['embedding'
        ] = new_embeddings
    model.params = params
    return model
