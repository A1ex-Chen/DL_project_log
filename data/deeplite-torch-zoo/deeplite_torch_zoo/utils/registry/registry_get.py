def get(self, registry_name: str) ->Registry:
    if registry_name not in self.registries:
        raise RuntimeError(
            f'Cannot find registry with registry_name "{registry_name}"')
    return self.registries[registry_name]
