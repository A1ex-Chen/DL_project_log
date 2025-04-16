def __setattr__(self, key, val):
    if key in self._RENAMED:
        log_first_n(logging.WARNING, "Metadata '{}' was renamed to '{}'!".
            format(key, self._RENAMED[key]), n=10)
        setattr(self, self._RENAMED[key], val)
    try:
        oldval = getattr(self, key)
        assert oldval == val, """Attribute '{}' in the metadata of '{}' cannot be set to a different value!
{} != {}""".format(
            key, self.name, oldval, val)
    except AttributeError:
        super().__setattr__(key, val)
