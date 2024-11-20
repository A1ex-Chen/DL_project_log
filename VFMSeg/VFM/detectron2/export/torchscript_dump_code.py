def dump_code(prefix, mod):
    code = get_code(mod)
    name = prefix or 'root model'
    if code is None:
        f.write(f'Could not found code for {name} (type={mod.original_name})\n'
            )
        f.write('\n')
    else:
        f.write(f'\nCode for {name}, type={mod.original_name}:\n')
        f.write(code)
        f.write('\n')
        f.write('-' * 80)
    for name, m in mod.named_children():
        dump_code(prefix + '.' + name, m)
