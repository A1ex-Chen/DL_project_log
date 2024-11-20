def register_extension(self, extension: str, clazz):
    already_registered_class = self._registry.get(extension, None)
    if (already_registered_class and already_registered_class.__module__ !=
        clazz.__module__):
        raise RuntimeError(
            f'Conflicting extension {self._name}/{extension}; {already_registered_class.__module__}.{already_registered_class.__name} and {clazz.__module__}.{clazz.__name__}'
            )
    elif already_registered_class is None:
        clazz_full_name = (f'{clazz.__module__}.{clazz.__name__}' if clazz
             is not None else 'None')
        LOGGER.debug(
            f'Registering extension {self._name}/{extension}: {clazz_full_name}'
            )
        self._registry[extension] = clazz
