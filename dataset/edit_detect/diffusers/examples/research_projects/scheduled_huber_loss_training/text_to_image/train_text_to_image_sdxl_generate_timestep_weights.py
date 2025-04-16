def generate_timestep_weights(args, num_timesteps):
    weights = torch.ones(num_timesteps)
    num_to_bias = int(args.timestep_bias_portion * num_timesteps)
    if args.timestep_bias_strategy == 'later':
        bias_indices = slice(-num_to_bias, None)
    elif args.timestep_bias_strategy == 'earlier':
        bias_indices = slice(0, num_to_bias)
    elif args.timestep_bias_strategy == 'range':
        range_begin = args.timestep_bias_begin
        range_end = args.timestep_bias_end
        if range_begin < 0:
            raise ValueError(
                'When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero.'
                )
        if range_end > num_timesteps:
            raise ValueError(
                'When using the range strategy for timestep bias, you must provide an ending timestep smaller than the number of timesteps.'
                )
        bias_indices = slice(range_begin, range_end)
    else:
        return weights
    if args.timestep_bias_multiplier <= 0:
        return ValueError(
            'The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps. If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead. A timestep bias multiplier less than or equal to 0 is not allowed.'
            )
    weights[bias_indices] *= args.timestep_bias_multiplier
    weights /= weights.sum()
    return weights
