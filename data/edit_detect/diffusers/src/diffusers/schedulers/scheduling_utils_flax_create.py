@classmethod
def create(cls, scheduler):
    config = scheduler.config
    if config.trained_betas is not None:
        betas = jnp.asarray(config.trained_betas, dtype=scheduler.dtype)
    elif config.beta_schedule == 'linear':
        betas = jnp.linspace(config.beta_start, config.beta_end, config.
            num_train_timesteps, dtype=scheduler.dtype)
    elif config.beta_schedule == 'scaled_linear':
        betas = jnp.linspace(config.beta_start ** 0.5, config.beta_end ** 
            0.5, config.num_train_timesteps, dtype=scheduler.dtype) ** 2
    elif config.beta_schedule == 'squaredcos_cap_v2':
        betas = betas_for_alpha_bar(config.num_train_timesteps, dtype=
            scheduler.dtype)
    else:
        raise NotImplementedError(
            f'beta_schedule {config.beta_schedule} is not implemented for scheduler {scheduler.__class__.__name__}'
            )
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    return cls(alphas=alphas, betas=betas, alphas_cumprod=alphas_cumprod)
