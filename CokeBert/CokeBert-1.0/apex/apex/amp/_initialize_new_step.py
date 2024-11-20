def new_step(*args, **kwargs):
    with disable_casts():
        output = old_step(*args, **kwargs)
    return output
