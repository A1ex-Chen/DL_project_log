def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
    decay_op = self._decay_weights_sparse_op(var, indices, apply_state=
        apply_state)
    with tf.control_dependencies([decay_op]):
        return super()._resource_apply_sparse(grad, var, indices,
            apply_state=apply_state)
