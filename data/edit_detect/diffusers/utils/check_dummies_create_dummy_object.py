def create_dummy_object(name, backend_name):
    """Create the code for the dummy object corresponding to `name`."""
    if name.isupper():
        return DUMMY_CONSTANT.format(name)
    elif name.islower():
        return DUMMY_FUNCTION.format(name, backend_name)
    else:
        return DUMMY_CLASS.format(name, backend_name)
