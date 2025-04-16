def set_builtin(name, value):
    if isinstance(__builtins__, dict):
        __builtins__[name] = value
    else:
        setattr(__builtins__, name, value)
