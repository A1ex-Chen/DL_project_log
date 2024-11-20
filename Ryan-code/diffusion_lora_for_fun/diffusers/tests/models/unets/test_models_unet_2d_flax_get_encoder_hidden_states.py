def get_encoder_hidden_states(self, seed=0, shape=(4, 77, 768), fp16=False):
    dtype = jnp.bfloat16 if fp16 else jnp.float32
    hidden_states = jnp.array(load_hf_numpy(self.get_file_format(seed,
        shape)), dtype=dtype)
    return hidden_states
