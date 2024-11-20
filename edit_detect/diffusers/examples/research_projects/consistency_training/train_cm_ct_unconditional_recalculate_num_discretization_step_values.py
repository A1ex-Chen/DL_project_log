def recalculate_num_discretization_step_values(discretization_steps, skip_steps
    ):
    """
        Recalculates all quantities depending on the number of discretization steps N.
        """
    noise_scheduler = CMStochasticIterativeScheduler(num_train_timesteps=
        discretization_steps, sigma_min=args.sigma_min, sigma_max=args.
        sigma_max, rho=args.rho)
    current_timesteps = get_karras_sigmas(discretization_steps, args.
        sigma_min, args.sigma_max, args.rho)
    valid_teacher_timesteps_plus_one = current_timesteps[:len(
        current_timesteps) - skip_steps + 1]
    timestep_weights = get_discretized_lognormal_weights(
        valid_teacher_timesteps_plus_one, p_mean=args.p_mean, p_std=args.p_std)
    timestep_loss_weights = get_loss_weighting_schedule(
        valid_teacher_timesteps_plus_one)
    current_timesteps = current_timesteps.to(accelerator.device)
    timestep_weights = timestep_weights.to(accelerator.device)
    timestep_loss_weights = timestep_loss_weights.to(accelerator.device)
    return (noise_scheduler, current_timesteps, timestep_weights,
        timestep_loss_weights)
