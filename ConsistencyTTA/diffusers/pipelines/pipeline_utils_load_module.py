def load_module(name, value):
    if value[0] is None:
        return False
    if name in passed_class_obj and passed_class_obj[name] is None:
        return False
    return True
