def get_project_root() ->Path:
    return Path(deeplite_torch_zoo.__file__).parents[1]
