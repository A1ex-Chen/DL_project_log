def dump_torchscript_IR(model, dir):
    """
    Dump IR of a TracedModule/ScriptModule/Function in various format (code, graph,
    inlined graph). Useful for debugging.

    Args:
        model (TracedModule/ScriptModule/ScriptFUnction): traced or scripted module
        dir (str): output directory to dump files.
    """
    dir = os.path.expanduser(dir)
    PathManager.mkdirs(dir)

    def _get_script_mod(mod):
        if isinstance(mod, torch.jit.TracedModule):
            return mod._actual_script_module
        return mod
    with PathManager.open(os.path.join(dir, 'model_ts_code.txt'), 'w') as f:

        def get_code(mod):
            try:
                return _get_script_mod(mod)._c.code
            except AttributeError:
                pass
            try:
                return mod.code
            except AttributeError:
                return None

        def dump_code(prefix, mod):
            code = get_code(mod)
            name = prefix or 'root model'
            if code is None:
                f.write(
                    f'Could not found code for {name} (type={mod.original_name})\n'
                    )
                f.write('\n')
            else:
                f.write(f'\nCode for {name}, type={mod.original_name}:\n')
                f.write(code)
                f.write('\n')
                f.write('-' * 80)
            for name, m in mod.named_children():
                dump_code(prefix + '.' + name, m)
        if isinstance(model, torch.jit.ScriptFunction):
            f.write(get_code(model))
        else:
            dump_code('', model)

    def _get_graph(model):
        try:
            return _get_script_mod(model)._c.dump_to_str(True, False, False)
        except AttributeError:
            return model.graph.str()
    with PathManager.open(os.path.join(dir, 'model_ts_IR.txt'), 'w') as f:
        f.write(_get_graph(model))
    with PathManager.open(os.path.join(dir, 'model_ts_IR_inlined.txt'), 'w'
        ) as f:
        f.write(str(model.inlined_graph))
    if not isinstance(model, torch.jit.ScriptFunction):
        with PathManager.open(os.path.join(dir, 'model.txt'), 'w') as f:
            f.write(str(model))
