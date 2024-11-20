def create_optimization_profiles(builder, inputs, batch_sizes=[1, 8, 16, 32,
    64]):
    if all([(inp.shape[0] > -1) for inp in inputs]):
        profile = builder.create_optimization_profile()
        for inp in inputs:
            fbs, shape = inp.shape[0], inp.shape[1:]
            profile.set_shape(inp.name, min=(fbs, *shape), opt=(fbs, *shape
                ), max=(fbs, *shape))
            return [profile]
    profiles = {}
    for bs in batch_sizes:
        if not profiles.get(bs):
            profiles[bs] = builder.create_optimization_profile()
        for inp in inputs:
            shape = inp.shape[1:]
            if inp.shape[0] > -1:
                bs = inp.shape[0]
            profiles[bs].set_shape(inp.name, min=(bs, *shape), opt=(bs, *
                shape), max=(bs, *shape))
    return list(profiles.values())
