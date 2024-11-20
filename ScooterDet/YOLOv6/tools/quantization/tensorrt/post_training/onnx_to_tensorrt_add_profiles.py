def add_profiles(config, inputs, opt_profiles):
    logger.debug('=== Optimization Profiles ===')
    for i, profile in enumerate(opt_profiles):
        for inp in inputs:
            _min, _opt, _max = profile.get_shape(inp.name)
            logger.debug('{} - OptProfile {} - Min {} Opt {} Max {}'.format
                (inp.name, i, _min, _opt, _max))
        config.add_optimization_profile(profile)
