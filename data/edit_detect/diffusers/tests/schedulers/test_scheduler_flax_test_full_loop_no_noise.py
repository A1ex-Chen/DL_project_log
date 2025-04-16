def test_full_loop_no_noise(self):
    sample = self.full_loop()
    result_sum = jnp.sum(jnp.abs(sample))
    result_mean = jnp.mean(jnp.abs(sample))
    if jax_device == 'tpu':
        assert abs(result_sum - 198.1275) < 0.01
        assert abs(result_mean - 0.258) < 0.001
    else:
        assert abs(result_sum - 198.1318) < 0.01
        assert abs(result_mean - 0.258) < 0.001
