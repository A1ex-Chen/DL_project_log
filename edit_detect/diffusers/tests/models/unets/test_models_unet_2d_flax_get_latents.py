def get_latents(self, seed=0, shape=(4, 4, 64, 64), fp16=False):
    dtype = jnp.bfloat16 if fp16 else jnp.float32
    image = jnp.array(load_hf_numpy(self.get_file_format(seed, shape)),
        dtype=dtype)
    return image
