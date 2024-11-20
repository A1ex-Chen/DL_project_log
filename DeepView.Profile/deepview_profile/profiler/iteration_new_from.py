@classmethod
def new_from(cls, model_provider, input_provider, iteration_provider,
    path_to_entry_point_dir, project_root):
    with user_code_environment(path_to_entry_point_dir, project_root):
        model = model_provider()
        iteration = iteration_provider(model)
    return cls(iteration, input_provider, path_to_entry_point_dir, project_root
        )
