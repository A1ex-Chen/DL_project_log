def set_func_save(handle, mod, fn, new_fn):
    cur_fn = get_func(mod, fn)
    handle._save_func(mod, fn, cur_fn)
    set_func(mod, fn, new_fn)
