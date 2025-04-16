def add_dependencies_load_info(load_info_dict, my_model):
    for model_name, model_dep in my_model.model_dependencies.models.items():
        if model_name not in load_info_dict:
            load_info_dict[model_name] = {'time': model_dep._load_time or 0,
                'memory_increment': model_dep._load_memory_increment or 0}
            add_dependencies_load_info(load_info_dict, model_dep)
