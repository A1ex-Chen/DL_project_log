def preprocess_mask(mask, dtype):
    w, h = mask.size
    w, h = (x - x % 32 for x in (w, h))
    mask = mask.resize((w, h))
    mask = jnp.array(mask.convert('L')).astype(dtype) / 255.0
    mask = jnp.expand_dims(mask, axis=(0, 1))
    return mask
