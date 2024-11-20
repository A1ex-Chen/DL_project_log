def patchClass(cls):
    for f in dir(cls):
        if isfunc(cls, f):
            add_wrapper(cls, f)
