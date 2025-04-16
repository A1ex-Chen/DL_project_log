def _ep(**kwargs):
    params = replace(m.params, **kwargs)
    return m.constructor(arch=m.arch, **asdict(params))
