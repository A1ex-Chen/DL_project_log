def __init__(self):
    for name in RNN_NAMES:
        for suffix in ['', '_cell']:
            fn_name = name + suffix
            setattr(self, fn_name, _gen_VF_wrapper(fn_name))
