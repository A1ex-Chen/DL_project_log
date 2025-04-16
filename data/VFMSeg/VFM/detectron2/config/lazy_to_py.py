@staticmethod
def to_py(cfg, prefix: str='cfg.'):
    """
        Try to convert a config object into Python-like psuedo code.

        Note that perfect conversion is not always possible. So the returned
        results are mainly meant to be human-readable, and not meant to be executed.

        Args:
            cfg: an omegaconf config object
            prefix: root name for the resulting code (default: "cfg.")


        Returns:
            str of formatted Python code
        """
    import black
    cfg = OmegaConf.to_container(cfg, resolve=True)

    def _to_str(obj, prefix=None, inside_call=False):
        if prefix is None:
            prefix = []
        if isinstance(obj, abc.Mapping) and '_target_' in obj:
            target = _convert_target_to_string(obj.pop('_target_'))
            args = []
            for k, v in sorted(obj.items()):
                args.append(f'{k}={_to_str(v, inside_call=True)}')
            args = ', '.join(args)
            call = f'{target}({args})'
            return ''.join(prefix) + call
        elif isinstance(obj, abc.Mapping) and not inside_call:
            key_list = []
            for k, v in sorted(obj.items()):
                if isinstance(v, abc.Mapping) and '_target_' not in v:
                    key_list.append(_to_str(v, prefix=prefix + [k + '.']))
                else:
                    key = ''.join(prefix) + k
                    key_list.append(f'{key}={_to_str(v)}')
            return '\n'.join(key_list)
        elif isinstance(obj, abc.Mapping):
            return '{' + ','.join(
                f'{repr(k)}: {_to_str(v, inside_call=inside_call)}' for k,
                v in sorted(obj.items())) + '}'
        elif isinstance(obj, list):
            return '[' + ','.join(_to_str(x, inside_call=inside_call) for x in
                obj) + ']'
        else:
            return repr(obj)
    py_str = _to_str(cfg, prefix=[prefix])
    try:
        return black.format_str(py_str, mode=black.Mode())
    except black.InvalidInput:
        return py_str
