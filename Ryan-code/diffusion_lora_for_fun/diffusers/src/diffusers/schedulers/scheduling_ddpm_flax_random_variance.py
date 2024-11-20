def random_variance():
    split_key = jax.random.split(key, num=1)[0]
    noise = jax.random.normal(split_key, shape=model_output.shape, dtype=
        self.dtype)
    return self._get_variance(state, t, predicted_variance=predicted_variance
        ) ** 0.5 * noise
