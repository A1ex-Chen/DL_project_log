def loss_and_grad(grad_idx, train_rng):
    minibatch = get_minibatch(batch, grad_idx
        ) if grad_idx is not None else batch
    sample_rng, train_rng = jax.random.split(train_rng, 2)
    loss, grad = grad_fn(state.params, minibatch, sample_rng)
    return loss, grad, train_rng
