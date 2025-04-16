def dump_to_str(self, obj, **kwargs):
    kwargs.setdefault('Dumper', Dumper)
    return yaml.dump(obj, **kwargs)
