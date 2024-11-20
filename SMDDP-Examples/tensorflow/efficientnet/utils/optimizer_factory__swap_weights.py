@tf.function
def _swap_weights(self):

    def fn_0(a, b):
        a.assign_add(b)
        return a

    def fn_1(b, a):
        b.assign(a - b)
        return b

    def fn_2(a, b):
        a.assign_sub(b)
        return a

    def swap(strategy, a_and_b):
        """Swap `a` and `b` and mirror to all devices."""
        for a, b in a_and_b:
            strategy.extended.update(a, fn_0, args=(b,))
            strategy.extended.update(b, fn_1, args=(a,))
            strategy.extended.update(a, fn_2, args=(b,))
    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(swap, args=(zip(self._average_weights, self.
        _model_weights),))
