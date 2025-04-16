def to_tf_var_name(name: str):
    for patt, repl in iter(var_map):
        name = name.replace(patt, repl)
    return 'bert/{}'.format(name)
