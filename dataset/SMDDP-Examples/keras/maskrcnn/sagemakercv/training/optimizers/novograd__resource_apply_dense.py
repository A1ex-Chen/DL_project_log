def _resource_apply_dense(self, grad, var, apply_state=None):
    var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
    weight_decay = self._get_hyper('weight_decay')
    grad_averaging = self._get_hyper('grad_averaging')
    v = self.get_slot(var, 'v')
    g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
    v_t = tf.cond(tf.equal(self.iterations, 0), lambda : g_2, lambda : v *
        coefficients['beta_2_t'] + g_2 * coefficients['one_minus_beta_2_t'])
    v_t = v.assign(v_t, use_locking=self._use_locking)
    if self.amsgrad:
        vhat = self.get_slot(var, 'vhat')
        vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self.
            _use_locking)
        grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
    else:
        grad = grad / (tf.sqrt(v_t) + self.epsilon)
    var_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(var_name):
        grad += weight_decay * var
    grad = tf.cond(tf.logical_and(grad_averaging, tf.not_equal(self.
        iterations, 0)), lambda : grad * coefficients['one_minus_beta_1_t'],
        lambda : grad)
    m = self.get_slot(var, 'm')
    return tf.raw_ops.ResourceApplyKerasMomentum(var=var.handle, accum=m.
        handle, lr=coefficients['lr_t'], grad=grad, momentum=coefficients[
        'beta_1_t'], use_locking=self._use_locking, use_nesterov=False)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
    coefficients = (apply_state or {}).get((var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
    weight_decay = self._get_hyper('weight_decay')
    grad_averaging = self._get_hyper('grad_averaging')
    v = self.get_slot(var, 'v')
    g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
    v_t = tf.cond(tf.equal(self.iterations, 0), lambda : g_2, lambda : v *
        coefficients['beta_2_t'] + g_2 * coefficients['one_minus_beta_2_t'])
    v_t = v.assign(v_t, use_locking=self._use_locking)
    if self.amsgrad:
        vhat = self.get_slot(var, 'vhat')
        vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self.
            _use_locking)
        grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
    else:
        grad = grad / (tf.sqrt(v_t) + self.epsilon)
    var_name = self._get_variable_name(var.name)
    if self._do_use_weight_decay(var_name):
        grad += weight_decay * tf.gather(var, indices)
    grad = tf.cond(tf.logical_and(grad_averaging, tf.not_equal(self.
        iterations, 0)), lambda : grad * coefficients['one_minus_beta_1_t'],
        lambda : grad)
    m = self.get_slot(var, 'm')
    return tf.raw_ops.ResourceSparseApplyKerasMomentum(var=var.handle,
        accum=m.handle, lr=coefficients['lr_t'], grad=grad, indices=indices,
        momentum=coefficients['beta_1_t'], use_locking=self._use_locking,
        use_nesterov=False)
