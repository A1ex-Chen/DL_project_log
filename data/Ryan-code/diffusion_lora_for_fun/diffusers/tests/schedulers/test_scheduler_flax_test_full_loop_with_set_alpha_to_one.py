def test_full_loop_with_set_alpha_to_one(self):
    sample = self.full_loop(set_alpha_to_one=True, beta_start=0.01)
    result_sum = jnp.sum(jnp.abs(sample))
    result_mean = jnp.mean(jnp.abs(sample))
    if jax_device == 'tpu':
        assert abs(result_sum - 186.83226) < 0.01
        assert abs(result_mean - 0.24327) < 0.001
    else:
        assert abs(result_sum - 186.9466) < 0.01
        assert abs(result_mean - 0.24342) < 0.001
