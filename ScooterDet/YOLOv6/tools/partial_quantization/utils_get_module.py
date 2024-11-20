def get_module(model, submodule_key):
    sub_tokens = submodule_key.split('.')
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    return cur_mod
