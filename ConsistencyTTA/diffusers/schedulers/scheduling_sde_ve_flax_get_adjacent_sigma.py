def get_adjacent_sigma(self, state, timesteps, t):
    return jnp.where(timesteps == 0, jnp.zeros_like(t), state.
        discrete_sigmas[timesteps - 1])
