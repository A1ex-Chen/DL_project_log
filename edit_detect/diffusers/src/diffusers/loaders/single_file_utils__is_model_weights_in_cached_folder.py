def _is_model_weights_in_cached_folder(cached_folder, name):
    pretrained_model_name_or_path = os.path.join(cached_folder, name)
    weights_exist = False
    for weights_name in [WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME]:
        if os.path.isfile(os.path.join(pretrained_model_name_or_path,
            weights_name)):
            weights_exist = True
    return weights_exist
