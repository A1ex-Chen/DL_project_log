def create_entrypoint(m: Model):

    def _ep(**kwargs):
        params = replace(m.params, **kwargs)
        return m.constructor(arch=m.arch, **asdict(params))
    return _ep
