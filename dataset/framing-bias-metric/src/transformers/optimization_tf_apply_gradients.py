def apply_gradients(self, grads_and_vars, name=None, **kwargs):
    grads, tvars = list(zip(*grads_and_vars))
    return super(AdamWeightDecay, self).apply_gradients(zip(grads, tvars),
        name=name, **kwargs)
