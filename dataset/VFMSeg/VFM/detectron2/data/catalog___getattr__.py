def __getattr__(self, key):
    if key in self._RENAMED:
        log_first_n(logging.WARNING, "Metadata '{}' was renamed to '{}'!".
            format(key, self._RENAMED[key]), n=10)
        return getattr(self, self._RENAMED[key])
    if len(self.__dict__) > 1:
        raise AttributeError(
            "Attribute '{}' does not exist in the metadata of dataset '{}'. Available keys are {}."
            .format(key, self.name, str(self.__dict__.keys())))
    else:
        raise AttributeError(
            f"Attribute '{key}' does not exist in the metadata of dataset '{self.name}': metadata is empty."
            )
