@classmethod
def _best_guess_weight_name(cls, pretrained_model_name_or_path_or_dict,
    file_extension='.safetensors', local_files_only=False):
    if local_files_only or HF_HUB_OFFLINE:
        raise ValueError(
            'When using the offline mode, you must specify a `weight_name`.')
    targeted_files = []
    if os.path.isfile(pretrained_model_name_or_path_or_dict):
        return
    elif os.path.isdir(pretrained_model_name_or_path_or_dict):
        targeted_files = [f for f in os.listdir(
            pretrained_model_name_or_path_or_dict) if f.endswith(
            file_extension)]
    else:
        files_in_repo = model_info(pretrained_model_name_or_path_or_dict
            ).siblings
        targeted_files = [f.rfilename for f in files_in_repo if f.rfilename
            .endswith(file_extension)]
    if len(targeted_files) == 0:
        return
    unallowed_substrings = {'scheduler', 'optimizer', 'checkpoint'}
    targeted_files = list(filter(lambda x: all(substring not in x for
        substring in unallowed_substrings), targeted_files))
    if any(f.endswith(LORA_WEIGHT_NAME) for f in targeted_files):
        targeted_files = list(filter(lambda x: x.endswith(LORA_WEIGHT_NAME),
            targeted_files))
    elif any(f.endswith(LORA_WEIGHT_NAME_SAFE) for f in targeted_files):
        targeted_files = list(filter(lambda x: x.endswith(
            LORA_WEIGHT_NAME_SAFE), targeted_files))
    if len(targeted_files) > 1:
        raise ValueError(
            f"Provided path contains more than one weights file in the {file_extension} format. Either specify `weight_name` in `load_lora_weights` or make sure there's only one  `.safetensors` or `.bin` file in  {pretrained_model_name_or_path_or_dict}."
            )
    weight_name = targeted_files[0]
    return weight_name
