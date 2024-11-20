def dump_to_fileobj(self, obj, file, **kwargs):
    kwargs.setdefault('Dumper', Dumper)
    yaml.dump(obj, file, **kwargs)
