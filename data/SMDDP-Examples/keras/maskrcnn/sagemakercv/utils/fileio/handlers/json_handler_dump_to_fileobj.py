def dump_to_fileobj(self, obj, file, **kwargs):
    obj = json.dumps(obj, cls=NumpyEncoder)
    json.dump(obj, file, **kwargs)
