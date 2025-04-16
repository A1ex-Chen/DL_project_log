def load_from_file(file_path, label, target):
    spec = importlib.util.spec_from_file_location(name=label, location=
        file_path)
    my_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(my_module)
    return getattr(my_module, target, None)
