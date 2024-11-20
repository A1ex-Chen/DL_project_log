def _prepare_local(self, var_device, var_dtype, apply_state):
    super()._prepare_local(var_device, var_dtype, apply_state)
    beta_1_t = tf.identity(self._get_hyper('beta_1', var_dtype))
    beta_2_t = tf.identity(self._get_hyper('beta_2', var_dtype))
    apply_state[var_device, var_dtype].update(dict(epsilon=tf.
        convert_to_tensor(self.epsilon, var_dtype), beta_1_t=beta_1_t,
        beta_2_t=beta_2_t, one_minus_beta_2_t=1 - beta_2_t,
        one_minus_beta_1_t=1 - beta_1_t))
