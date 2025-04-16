def call(self, inputs, mode='embedding'):
    if self._abs_scope_name is None:
        return self._layer.call(inputs, mode)
    with tf.compat.v1.variable_scope(self._abs_scope_name,
        auxiliary_name_scope=False) as abs_scope_name:
        with tf.name_scope(abs_scope_name.original_name_scope):
            return self._layer.call(inputs, mode)
