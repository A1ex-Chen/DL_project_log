def jax_cosine_distance(emb_1, emb_2, eps=1e-12):
    norm_emb_1 = jnp.divide(emb_1.T, jnp.clip(jnp.linalg.norm(emb_1, axis=1
        ), a_min=eps)).T
    norm_emb_2 = jnp.divide(emb_2.T, jnp.clip(jnp.linalg.norm(emb_2, axis=1
        ), a_min=eps)).T
    return jnp.matmul(norm_emb_1, norm_emb_2.T)
