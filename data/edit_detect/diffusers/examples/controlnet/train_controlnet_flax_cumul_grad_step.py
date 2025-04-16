def cumul_grad_step(grad_idx, loss_grad_rng):
    cumul_loss, cumul_grad, train_rng = loss_grad_rng
    loss, grad, new_train_rng = loss_and_grad(grad_idx, train_rng)
    cumul_loss, cumul_grad = jax.tree_map(jnp.add, (cumul_loss, cumul_grad),
        (loss, grad))
    return cumul_loss, cumul_grad, new_train_rng
