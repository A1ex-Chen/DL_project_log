def __init__(self, registry_list):
    regs = [r for r in registry_list if isinstance(r, Registry)]
    self.registries = {}
    for r in regs:
        if r.name in self.registries:
            raise RuntimeError(
                f'There are more than one registry with the name "{r.name}"')
        self.registries[r.name] = r
