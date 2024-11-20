@staticmethod
def load(filename: str, keys: Union[None, str, Tuple[str, ...]]=None):
    """
        Load a config file.

        Args:
            filename: absolute path or relative path w.r.t. the current working directory
            keys: keys to load and return. If not given, return all keys
                (whose values are config objects) in a dict.
        """
    has_keys = keys is not None
    filename = filename.replace('/./', '/')
    if os.path.splitext(filename)[1] not in ['.py', '.yaml', '.yml']:
        raise ValueError(
            f'Config file {filename} has to be a python or yaml file.')
    if filename.endswith('.py'):
        _validate_py_syntax(filename)
        with _patch_import():
            module_namespace = {'__file__': filename, '__package__':
                _random_package_name(filename)}
            with PathManager.open(filename) as f:
                content = f.read()
            exec(compile(content, filename, 'exec'), module_namespace)
        ret = module_namespace
    else:
        with PathManager.open(filename) as f:
            obj = yaml.unsafe_load(f)
        ret = OmegaConf.create(obj, flags={'allow_objects': True})
    if has_keys:
        if isinstance(keys, str):
            return _cast_to_config(ret[keys])
        else:
            return tuple(_cast_to_config(ret[a]) for a in keys)
    else:
        if filename.endswith('.py'):
            ret = DictConfig({name: _cast_to_config(value) for name, value in
                ret.items() if isinstance(value, (DictConfig, ListConfig,
                dict)) and not name.startswith('_')}, flags={
                'allow_objects': True})
        return ret
