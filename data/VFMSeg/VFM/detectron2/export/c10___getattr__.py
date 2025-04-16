def __getattr__(self, name):
    if name not in self.batch_extra_fields:
        raise AttributeError("Cannot find field '{}' in the given Instances!"
            .format(name))
    return self.batch_extra_fields[name]
