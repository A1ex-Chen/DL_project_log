def _resource_apply_dense(self, grad, var, apply_state=None):
    with tf.control_dependencies([self._decay_weights_op(var, apply_state=
        apply_state)]):
        return super()._resource_apply_dense(grad, var, apply_state=apply_state
            )
